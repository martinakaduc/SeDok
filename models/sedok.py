import logging
import math
import os
from datetime import datetime
from turtle import forward

import dgl
import torch
from torch import nn
from torch.distributions.uniform import Uniform
from dgl import function as fn

from commons.process_mols import AtomEncoder, rec_atom_feature_dims, rec_residue_feature_dims, lig_feature_dims
from commons.logger import log


class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


def normalize(v):
    return v / torch.linalg.norm(v)


def find_additional_vertical_vector(vector, ez=None, device='cpu'):
    look_at_vector = normalize(vector)

    # ez = torch.FloatTensor([0, 0, 1]).to(device)
    if ez is None:
        ez = normalize(Uniform(-1,1).sample((3,)).to(device))

    while torch.isclose(abs(look_at_vector@ez), torch.ones(1).to(device), atol=5e-3):
        ez = normalize(Uniform(-1,1).sample((3,)).to(device))

    up_vector = normalize(ez - torch.dot(look_at_vector, ez) * look_at_vector)
    return up_vector, ez


def calc_rotation_matrix(v1_start, v2_start, v1_target, v2_target):
    """
    calculating M the rotation matrix from base U to base V
    M @ U = V
    M = V @ U^-1
    """
    u1_start = normalize(v1_start)
    u2_start = normalize(v2_start)
    u3_start = normalize(torch.cross(u1_start, u2_start))

    u1_target = normalize(v1_target)
    u2_target = normalize(v2_target)
    u3_target = normalize(torch.cross(u1_target, u2_target))

    U = torch.hstack([u1_start.view(3, 1), u2_start.view(3, 1), u3_start.view(3, 1)])
    V = torch.hstack([u1_target.view(3, 1), u2_target.view(3, 1), u3_target.view(3, 1)])

    return V@torch.linalg.inv(U)


def get_rotation_matrix(start_look_at_vector, target_look_at_vector, reference_vector=None, start_up_vector=None, target_up_vector=None, device='cpu'):
    # if reference_vector is None:
    #     reference_vector = Uniform(-1,1).sample((3,)).to(device)
    # EZ must be normalized
    start_ez = target_ez = torch.zeros(1).to(device)

    if start_up_vector is None:
        start_up_vector, start_ez = find_additional_vertical_vector(start_look_at_vector, ez=reference_vector, device=device)

    if target_up_vector is None:
        target_up_vector, target_ez = find_additional_vertical_vector(target_look_at_vector, ez=reference_vector, device=device)

    rot_mat = calc_rotation_matrix(start_look_at_vector, start_up_vector, target_look_at_vector, target_up_vector)
    return rot_mat, torch.exp((normalize(start_look_at_vector)@start_ez.T)**2) + torch.exp((normalize(target_look_at_vector)@target_ez.T)**2) - 2


def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    elif type == 'relu':
        return nn.ReLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0' or layer_norm_type == 0
        return nn.Identity()


def apply_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)


def cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return attention_x


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols, device=device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask

def get_base_loss(base):
    u1, u2, u3 = base
    return abs(u1 @ u2.T) + abs(u2 @ u3.T) + abs(u3 @ u1.T)

def get_vector_loss(vectors, mean=5, std=2):
    return torch.sum(1 - torch.exp(-((torch.linalg.vector_norm(vectors, dim=1) - mean)**2) /std))

class CoordsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coords):
        norm = coords.norm(dim=-1, keepdim=True)
        normed_coords = coords / norm.clamp(min=self.eps)
        return normed_coords * self.scale

# =================================================================================================================
class ISET_Layer(nn.Module):
    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            lig_input_edge_feats_dim,
            rec_input_edge_feats_dim,
            nonlin,
            cross_msgs,
            layer_norm,
            layer_norm_coords,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            leakyrelu_neg_slope,
            debug,
            device,
            dropout,
            save_trajectories=False,
            rec_square_distance_scale=1,
            standard_norm_order=False,
            normalize_coordinate_update=False,
            lig_evolve=True,
            rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = False,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            lig_no_softmax=False,
            rec_no_softmax=False,
            nhop=None
    ):

        super(ISET_Layer, self).__init__()

        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update =norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.debug = debug
        self.device = device
        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories
        self.lig_no_softmax = lig_no_softmax
        self.rec_no_softmax = rec_no_softmax
        self.nhop = nhop

        # EDGES
        lig_edge_mlp_input_dim = (h_feats_dim * 2) + lig_input_edge_feats_dim
        if self.use_dist_in_layers and self.lig_evolve:
            lig_edge_mlp_input_dim += len(self.all_sigmas_dist)
        if self.standard_norm_order:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
            )
        else:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
            )
        rec_edge_mlp_input_dim = (h_feats_dim * 2) + rec_input_edge_feats_dim
        if self.use_dist_in_layers and self.rec_evolve:
            rec_edge_mlp_input_dim += len(self.all_sigmas_dist)
        if self.standard_norm_order:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
            )
        else:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
            )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        # normalization of x_i - x_j is not currently used
        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)
        if self.fine_tune:
            if self.norm_cross_coords_update:
                self.lig_cross_coords_norm = CoordsNorm(scale_init=1e-2)
                self.rec_cross_coords_norm = CoordsNorm(scale_init=1e-2)
            else:
                self.lig_cross_coords_norm =nn.Identity()
                self.rec_cross_coords_norm = nn.Identity()

        self.att_mlp_Q_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        if self.standard_norm_order:
            self.node_mlp_lig = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim, h_feats_dim),
                get_layer_norm(layer_norm, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(h_feats_dim, out_feats_dim),
                get_layer_norm(layer_norm, out_feats_dim),
            )
        else:
            self.node_mlp_lig = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim, h_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, h_feats_dim),
                nn.Linear(h_feats_dim, out_feats_dim),
            )
        if self.standard_norm_order:
            self.node_mlp = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim, h_feats_dim),
                get_layer_norm(layer_norm, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(h_feats_dim, out_feats_dim),
                get_layer_norm(layer_norm, out_feats_dim),
            )
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim, h_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, h_feats_dim),
                nn.Linear(h_feats_dim, out_feats_dim),
            )

        self.final_h_layernorm_layer_lig = get_norm(self.final_h_layer_norm, out_feats_dim)
        self.final_h_layernorm_layer = get_norm(self.final_h_layer_norm, out_feats_dim)

        self.pre_crossmsg_norm_lig = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)

        self.post_crossmsg_norm_lig = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec = get_norm(self.post_crossmsg_norm_type, h_feats_dim)

        if self.standard_norm_order:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, 1)
            )
        else:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, 1)
            )
        if self.standard_norm_order:
            self.coords_mlp_rec = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, 1)
            )
        else:
            self.coords_mlp_rec = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, 1)
            )
        if self.fine_tune:
            self.att_mlp_cross_coors_Q = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )
            self.att_mlp_cross_coors_Q_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )
        
        # self.zero = torch.zeros(1).to(self.device)
        # if self.nhop:
        #     self.W_lig_local_attn = nn.Parameter(torch.zeros(size=(out_feats_dim, out_feats_dim)))
        #     self.W_rec_local_attn = nn.Parameter(torch.zeros(size=(out_feats_dim, out_feats_dim)))
        #     self.gate = nn.Linear(out_feats_dim * 2, 1)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges_lig(self, edges):
        if self.use_dist_in_layers and self.lig_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.lig_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.lig_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_rec(self, edges):
        if self.use_dist_in_layers and self.rec_evolve:
            squared_distance = torch.sum(edges.data['x_rel'] ** 2, dim=1, keepdim=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_mag = torch.cat([torch.exp(-(squared_distance / self.rec_square_distance_scale) / sigma) for sigma in
                                   self.all_sigmas_dist], dim=-1)
            return {'msg': self.rec_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.rec_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def update_x_moment_lig(self, edges):
        edge_coef_ligand = self.coords_mlp_lig(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.lig_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_ligand}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_lig_multihop(self, edges):    
        edge_coef_ligand = torch.sigmoid(self.coords_mlp_lig(edges.data['msg']))  # \phi^x(m_{i->j})
        return {'m': edges.src['x_now'] * edge_coef_ligand}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_rec(self, edges):
        edge_coef_rec = self.coords_mlp_rec(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.rec_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_rec}  # (x_i - x_j) * \phi^x(m_{i->j})
    
    def update_x_moment_rec_multihop(self, edges):
        edge_coef_rec = torch.sigmoid(self.coords_mlp_rec(edges.data['msg']))  # \phi^x(m_{i->j})
        return {'m': edges.src['x_now'] * edge_coef_rec}  # (x_i - x_j) * \phi^x(m_{i->j})

    def msg_x_with_e(self, edges):
        return {'z': edges.src['x_update'], 'e': edges.data['coord_attn']}

    def reduce_x_by_neighbors(self, nodes):
        alpha = torch.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'x_aggre': h}

    def add_x_aggre(self, nodes):
        return {'x_update': self.x_connection_init * nodes.data['x_now'] + (1. - self.x_connection_init) * nodes.data['x_aggre']}

    def forward(self, lig_graph, rec_graph, coords_lig, h_feats_lig, original_ligand_node_features, orig_coords_lig,
                coords_rec, h_feats_rec, original_receptor_node_features, orig_coords_rec, mask, geometry_graph):
        with lig_graph.local_scope() and rec_graph.local_scope():
            lig_graph.ndata['x_now'] = coords_lig
            rec_graph.ndata['x_now'] = coords_rec
            lig_graph.ndata['feat'] = h_feats_lig  # first time set here
            rec_graph.ndata['feat'] = h_feats_rec

            if self.debug:
                log(torch.max(lig_graph.ndata['x_now'].abs()), 'x_now : x_i at layer entrance')
                log(torch.max(lig_graph.ndata['feat'].abs()), 'data[feat] = h_i at layer entrance')

            if self.lig_evolve:
                lig_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))  # x_i - x_j
                if self.debug:
                    log(torch.max(lig_graph.edata['x_rel'].abs()), 'x_rel : x_i - x_j')
            if self.rec_evolve:
                rec_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))

            lig_graph.apply_edges(self.apply_edges_lig)  ## i->j edge:  [h_i h_j]
            rec_graph.apply_edges(self.apply_edges_rec)

            if self.debug:
                log(torch.max(lig_graph.edata['msg'].abs()),
                    'data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)')

            h_feats_lig_norm = apply_norm(lig_graph, h_feats_lig, self.final_h_layer_norm, self.final_h_layernorm_layer)
            h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.final_h_layer_norm, self.final_h_layernorm_layer)
            cross_attention_lig_feat = cross_attention(self.att_mlp_Q_lig(h_feats_lig_norm),
                                                       self.att_mlp_K(h_feats_rec_norm),
                                                       self.att_mlp_V(h_feats_rec_norm), mask, self.cross_msgs)
            cross_attention_rec_feat = cross_attention(self.att_mlp_Q(h_feats_rec_norm),
                                                       self.att_mlp_K_lig(h_feats_lig_norm),
                                                       self.att_mlp_V_lig(h_feats_lig_norm), mask.transpose(0, 1),
                                                       self.cross_msgs)
            cross_attention_lig_feat = apply_norm(lig_graph, cross_attention_lig_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer)
            cross_attention_rec_feat = apply_norm(rec_graph, cross_attention_rec_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer)

            if self.debug:
                log(torch.max(cross_attention_lig_feat.abs()), 'aggr_cross_msg(i) = sum_j a_{i,j} * h_j')

            if self.lig_evolve:
                # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
                if not self.nhop:
                    lig_graph.update_all(self.update_x_moment_lig, fn.mean('m', 'x_update'))
                    x_evolved_lig = self.x_connection_init * orig_coords_lig + (1. - self.x_connection_init) * \
                                    (lig_graph.ndata['x_now']) + lig_graph.ndata['x_update']
                else:
                    # lig_graph.update_all(self.update_x_moment_lig_multihop, fn.mean('m', 'x_update'))
                    # x_evolved_lig = lig_graph.ndata['x_update']
                    # for idx in range(self.nhop):
                    #     x_evolved_lig = self.x_connection_init * lig_graph.ndata['x_now'] + (1. - self.x_connection_init) * x_evolved_lig
                    # x_evolved_lig = self.x_connection_init * orig_coords_lig + (1. - self.x_connection_init) * x_evolved_lig
                    
                    lig_graph.apply_edges(lambda x: {"coord_attn": nn.functional.leaky_relu(self.coords_mlp_lig(x.data['msg']))})
                    lig_graph.apply_nodes(lambda nodes: {'x_update' : nodes.data['x_now']})
                    for idx in range(self.nhop):
                        lig_graph.update_all(self.msg_x_with_e, self.reduce_x_by_neighbors, self.add_x_aggre)
                    x_evolved_lig = lig_graph.ndata['x_update']
            else:
                x_evolved_lig = coords_lig

            if self.rec_evolve:
                if not self.nhop:
                    rec_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                    x_evolved_rec = self.x_connection_init * orig_coords_rec + (1. - self.x_connection_init) * \
                                    (rec_graph.ndata['x_now']) + rec_graph.ndata['x_update']
                else:
                    # rec_graph.update_all(self.update_x_moment_rec_multihop, fn.mean('m', 'x_update'))
                    # x_evolved_rec = rec_graph.ndata['x_update']
                    # for idx in range(self.nhop):
                    #     x_evolved_rec = self.x_connection_init * rec_graph.ndata['x_now'] + (1. - self.x_connection_init) * x_evolved_rec
                    # x_evolved_rec = self.x_connection_init * orig_coords_rec + (1. - self.x_connection_init) * x_evolved_rec
                    
                    rec_graph.apply_edges(lambda x: {"coord_attn": nn.functional.leaky_relu(self.coords_mlp_rec(x.data['msg']))})
                    rec_graph.apply_nodes(lambda nodes: {'x_update' : nodes.data['x_now']})
                    for idx in range(self.nhop):
                        rec_graph.update_all(self.msg_x_with_e, self.reduce_x_by_neighbors, self.add_x_aggre)
                    x_evolved_rec = rec_graph.ndata['x_update']
            else:
                x_evolved_rec = coords_rec

            if not self.nhop:
                lig_graph.update_all(fn.copy_edge('msg', 'm'), fn.mean('m', 'aggr_msg'))
                rec_graph.update_all(fn.copy_edge('msg', 'm'), fn.mean('m', 'aggr_msg'))
            else:
                for idx in range(self.nhop):
                    if idx == 0:
                        lig_graph.update_all(fn.copy_edge('msg', 'm'), fn.mean('m', 'msg'+str(idx)))
                        rec_graph.update_all(fn.copy_edge('msg', 'm'), fn.mean('m', 'msg'+str(idx)))
                    else:
                        def copy_n2e(edge):
                            return {'msg'+str(idx-1): edge.src['msg'+str(idx-1)]}
                        lig_graph.apply_edges(copy_n2e)
                        rec_graph.apply_edges(copy_n2e)

                        lig_graph.update_all(fn.copy_edge('msg'+str(idx-1), 'm'), fn.mean('m', 'msg'+str(idx)))
                        rec_graph.update_all(fn.copy_edge('msg'+str(idx-1), 'm'), fn.mean('m', 'msg'+str(idx)))
                
                lig_graph.ndata['aggr_msg'] = lig_graph.ndata['msg'+str(self.nhop-1)]
                rec_graph.ndata['aggr_msg'] = rec_graph.ndata['msg'+str(self.nhop-1)]

            if self.fine_tune:
                x_evolved_lig = x_evolved_lig + self.att_mlp_cross_coors_V_lig(h_feats_lig) * (
                        self.lig_cross_coords_norm(lig_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q_lig(h_feats_lig),
                                                                   self.att_mlp_cross_coors_K(h_feats_rec),
                                                                   rec_graph.ndata['x_now'], mask, self.cross_msgs)))
            if self.fine_tune:
                x_evolved_rec = x_evolved_rec + self.att_mlp_cross_coors_V(h_feats_rec) * (
                        self.rec_cross_coords_norm(rec_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q(h_feats_rec),
                                                                   self.att_mlp_cross_coors_K_lig(h_feats_lig),
                                                                   lig_graph.ndata['x_now'], mask.transpose(0, 1),
                                                                   self.cross_msgs)))
            trajectory = []
            if self.save_trajectories: trajectory.append(x_evolved_lig.detach().cpu())
            if self.loss_geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                geom_loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
            else:
                geom_loss = 0
            if self.geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps): # T
                    d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1) # d_z
                    Loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_lig[src] - x_evolved_lig[dst])
                    geometry_graph.edata['partial_grads'] = 2 * (d_squared - geometry_graph.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph.update_all(fn.copy_edge('partial_grads', 'partial_grads_msg'),
                                              fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph.ndata['grad_x_evolved']
                    x_evolved_lig = x_evolved_lig + self.geometry_reg_step_size * grad_x_evolved
                    if self.save_trajectories:
                        trajectory.append(x_evolved_lig.detach().cpu())

            if self.debug:
                log(torch.max(lig_graph.ndata['aggr_msg'].abs()), 'data[aggr_msg]: \sum_j m_{i->j}')
                if self.lig_evolve:
                    log(torch.max(lig_graph.ndata['x_update'].abs()),
                        'data[x_update] : \sum_j (x_i - x_j) * \phi^x(m_{i->j})')
                    log(torch.max(x_evolved_lig.abs()), 'x_i new = x_evolved_lig : x_i + data[x_update]')

            input_node_upd_ligand = torch.cat((self.node_norm(lig_graph.ndata['feat']),
                                               lig_graph.ndata['aggr_msg'],
                                               cross_attention_lig_feat,
                                               original_ligand_node_features), dim=-1)

            input_node_upd_receptor = torch.cat((self.node_norm(rec_graph.ndata['feat']),
                                                 rec_graph.ndata['aggr_msg'],
                                                 cross_attention_rec_feat,
                                                 original_receptor_node_features), dim=-1)

            # Skip connections 1-hop
            if self.h_feats_dim == self.out_feats_dim:
                node_upd_ligand = self.skip_weight_h * self.node_mlp_lig(input_node_upd_ligand) + (
                        1. - self.skip_weight_h) * h_feats_lig
                node_upd_receptor = self.skip_weight_h * self.node_mlp(input_node_upd_receptor) + (
                        1. - self.skip_weight_h) * h_feats_rec
            else:
                node_upd_ligand = self.node_mlp_lig(input_node_upd_ligand)
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            # Multi-hop
            # node_upd_ligand = self.node_mlp_lig(input_node_upd_ligand)
            # node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            # if self.h_feats_dim == self.out_feats_dim:
            #     lig_e = torch.einsum('jl,kl->jk', (torch.matmul(node_upd_ligand, self.W_lig_local_attn), node_upd_ligand))
            #     lig_e = lig_e + lig_e.T

            #     lig_adj = torch.eye(lig_graph.number_of_nodes()).to(self.device) + lig_graph.adj(ctx=self.device)
            #     lig_local_attention = torch.where(lig_adj > 0, lig_e, self.zero)
            #     lig_local_attention = torch.softmax(lig_local_attention, dim=1)

            #     # lig_z = node_upd_ligand
            #     for _ in range(self.nhop):
            #         lig_az = torch.relu(torch.einsum('ij,jk->ik',(lig_local_attention, node_upd_ligand)))
            #         lig_coeff = torch.sigmoid(self.gate(torch.cat([h_feats_lig, lig_az], -1)))
            #         node_upd_ligand = lig_coeff * h_feats_lig + (1 - lig_coeff) * lig_az

            #     rec_e = torch.einsum('jl,kl->jk', (torch.matmul(node_upd_receptor, self.W_rec_local_attn), node_upd_receptor))
            #     rec_e = rec_e + rec_e.T

            #     rec_adj = torch.eye(rec_graph.number_of_nodes()).to(self.device) + rec_graph.adj(ctx=self.device)
            #     rec_local_attention = torch.where(rec_adj > 0, rec_e, self.zero)
            #     rec_local_attention = torch.softmax(rec_local_attention, dim=1)

            #     # rec_z = node_upd_receptor
            #     for _ in range(self.nhop):
            #         rec_az = torch.relu(torch.einsum('ij,jk->ik',(rec_local_attention, node_upd_receptor)))
            #         rec_coeff = torch.sigmoid(self.gate(torch.cat([h_feats_rec, rec_az], -1)))
            #         node_upd_receptor = rec_coeff * h_feats_rec + (1 - rec_coeff) * rec_az

            if self.debug:
                log('node_mlp params')
                for p in self.node_mlp.parameters():
                    log(torch.max(p.abs()), 'max node_mlp_params')
                    log(torch.min(p.abs()), 'min of abs node_mlp_params')
                log(torch.max(input_node_upd_ligand.abs()), 'concat(h_i, aggr_msg, aggr_cross_msg)')
                log(torch.max(node_upd_ligand), 'h_i new = h_i + MLP(h_i, aggr_msg, aggr_cross_msg)')

            node_upd_ligand = apply_norm(lig_graph, node_upd_ligand, self.final_h_layer_norm,
                                         self.final_h_layernorm_layer_lig)
            node_upd_receptor = apply_norm(rec_graph, node_upd_receptor,
                                           self.final_h_layer_norm, self.final_h_layernorm_layer)

            # rotation_matrices = []
            # translation_vectors = []

            ##### CALCULATING ROTATION MATRIX AND TRANSLATION VECTOR #####
            
            # trajectory = []
            # if self.save_trajectories:
            #     trajectory.append(new_x_evolved_lig.detach().cpu())
            
            # if self.loss_geometry_regularization:
            #     src, dst = geometry_graph.edges()
            #     src = src.long()
            #     dst = dst.long()
            #     d_squared = torch.sum((new_x_evolved_lig[src] - new_x_evolved_lig[dst]) ** 2, dim=1)
            #     geom_loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
            # else:
            #     geom_loss = 0

        return x_evolved_lig, node_upd_ligand, x_evolved_rec, node_upd_receptor, trajectory, geom_loss#, rotation_matrices, translation_vectors

    def __repr__(self):
        return "ISET Layer " + str(self.__dict__)

# =================================================================================================================
class ISET(nn.Module):
    def __init__(self, n_lays, debug, device, use_rec_atoms, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, residue_emb_dim, lay_hid_dim, locknkey,
                 dropout, nonlin, leakyrelu_neg_slope, num_att_heads=0, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_lig_feats=None, move_keypts_back=False, normalize_Z_lig_directions=False,
                 unnormalized_rotation_weights=False, centroid_keypts_construction_rec=False,
                 centroid_keypts_construction_lig=False, rec_no_softmax=False, lig_no_softmax=False,
                 normalize_Z_rec_directions=False,
                 centroid_keypts_construction=False, evolve_only=False, separate_lig=False, save_trajectories=False, multihop_strategy=None, **kwargs):
        super(ISET, self).__init__()
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        self.unnormalized_rotation_weights = unnormalized_rotation_weights
        self.separate_lig =separate_lig
        self.use_rec_atoms = use_rec_atoms
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.move_keypts_back = move_keypts_back
        self.normalize_Z_lig_directions = normalize_Z_lig_directions
        self.centroid_keypts_construction = centroid_keypts_construction
        self.centroid_keypts_construction_rec = centroid_keypts_construction_rec
        self.centroid_keypts_construction_lig = centroid_keypts_construction_lig
        self.normalize_Z_rec_directions = normalize_Z_rec_directions
        self.rec_no_softmax = rec_no_softmax
        self.lig_no_softmax = lig_no_softmax
        self.evolve_only = evolve_only
        self.locknkey = locknkey
        if multihop_strategy:
            self.multihop_strategy = multihop_strategy
        else:
            self.multihop_strategy = [None] * n_lays

        assert self.locknkey in ["direct_base", "direct", "indirect"]

        self.lig_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                             feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
                                             n_feats_to_use=num_lig_feats)
        if self.separate_lig:
            self.lig_separate_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                                 feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
                                                 n_feats_to_use=num_lig_feats)
        if self.use_rec_atoms:
            self.rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_atom_feature_dims, use_scalar_feat=use_scalar_features)
        else:
            self.rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_residue_feature_dims, use_scalar_feat=use_scalar_features)

        input_node_feats_dim = residue_emb_dim
        if self.use_mean_node_features:
            input_node_feats_dim += 5  ### Additional features from mu_r_norm
        self.iset_layers = nn.ModuleList()
        self.iset_layers.append(
            ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=input_node_feats_dim,
                        out_feats_dim=lay_hid_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,
                        nhop=self.multihop_strategy[0], **kwargs))

        if shared_layers:
            interm_lay = ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                                     h_feats_dim=lay_hid_dim,
                                     out_feats_dim=lay_hid_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,
                                     nhop=self.multihop_strategy[0], **kwargs)
            for layer_idx in range(1, n_lays):
                self.iset_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.iset_layers.append(
                    ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                                h_feats_dim=lay_hid_dim,
                                out_feats_dim=lay_hid_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,
                                nhop=self.multihop_strategy[layer_idx], **kwargs))
        if self.separate_lig:
            self.iset_layers_separate = nn.ModuleList()
            self.iset_layers_separate.append(
                ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                            h_feats_dim=input_node_feats_dim,
                            out_feats_dim=lay_hid_dim,
                            nonlin=nonlin,
                            cross_msgs=self.cross_msgs,
                            leakyrelu_neg_slope=leakyrelu_neg_slope,
                            debug=debug,
                            device=device,
                            dropout=dropout,
                            save_trajectories=save_trajectories,
                            nhop=self.multihop_strategy[0], **kwargs))

            if shared_layers:
                interm_lay = ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                                         h_feats_dim=lay_hid_dim,
                                         out_feats_dim=lay_hid_dim,
                                         cross_msgs=self.cross_msgs,
                                         nonlin=nonlin,
                                         leakyrelu_neg_slope=leakyrelu_neg_slope,
                                         debug=debug,
                                         device=device,
                                         dropout=dropout,
                                         save_trajectories=save_trajectories,
                                         nhop=self.multihop_strategy[0], **kwargs)
                for layer_idx in range(1, n_lays):
                    self.iset_layers_separate.append(interm_lay)
            else:
                for layer_idx in range(1, n_lays):
                    debug_this_layer = debug if n_lays - 1 == layer_idx else False
                    self.iset_layers_separate.append(
                        ISET_Layer(orig_h_feats_dim=input_node_feats_dim,
                                    h_feats_dim=lay_hid_dim,
                                    out_feats_dim=lay_hid_dim,
                                    cross_msgs=self.cross_msgs,
                                    nonlin=nonlin,
                                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                                    debug=debug_this_layer,
                                    device=device,
                                    dropout=dropout,
                                    save_trajectories=save_trajectories,
                                    nhop=self.multihop_strategy[layer_idx], **kwargs))
        # Attention layers
        self.out_feats_dim = lay_hid_dim
        # self.reset_parameters()

        if self.normalize_Z_lig_directions:
            self.Z_lig_dir_norm = CoordsNorm()
        if self.normalize_Z_rec_directions:
            self.Z_rec_dir_norm = CoordsNorm()
    
        # Rotation matrix & translation vector
        # self.W_feature = nn.Linear(out_feats_dim, out_feats_dim, bias=False)
        # self.R_linear = nn.Linear(3, 3, bias=False)
        # self.R_norm = get_norm("LN", 3)
        # self.feat_2_coord = nn.Linear(out_feats_dim, 3, bias=False)

        self.h_mean_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )

        self.h_mean_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        
        self.keypts_attention_rec_trans = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim, bias=False))

        self.keypts_queries_rec_trans = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim, bias=False))

        if (self.locknkey == "direct") or (self.locknkey == "indirect"):
            self.num_att_heads = 1
        elif self.locknkey == "direct_base":
            self.num_att_heads = 2
        else:
            self.num_att_heads = num_att_heads

        self.keypts_attention_rec_rot = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim*self.num_att_heads, bias=False))

        self.keypts_queries_rec_rot = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim*self.num_att_heads, bias=False))

        self.keypts_attention_lig_rot = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim*self.num_att_heads, bias=False))

        self.keypts_queries_lig_rot = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim*self.num_att_heads, bias=False))

        if self.locknkey == "indirect":
            self.xy_mask = torch.Tensor([1, 1, 0]).to(self.device)
            self.yz_mask = torch.Tensor([0, 1, 1]).to(self.device)
            self.zx_mask = torch.Tensor([1, 0, 1]).to(self.device)

    def forward(self, lig_graph, rec_graph, geometry_graph, complex_names, epoch):
        orig_coords_lig = lig_graph.ndata['new_x']
        orig_coords_rec = rec_graph.ndata['x']

        coords_lig = lig_graph.ndata['new_x']
        coords_rec = rec_graph.ndata['x']

        h_feats_lig = self.lig_atom_embedder(lig_graph.ndata['feat'])

        if self.use_rec_atoms:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])
        else:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_lig = rand_dist.sample([h_feats_lig.size(0), self.random_vec_dim]).to(self.device)
        rand_h_rec = rand_dist.sample([h_feats_rec.size(0), self.random_vec_dim]).to(self.device)
        h_feats_lig = torch.cat([h_feats_lig, rand_h_lig], dim=1)
        h_feats_rec = torch.cat([h_feats_rec, rand_h_rec], dim=1)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'max h_feats_lig before layers and noise ')
            log(torch.max(h_feats_rec.abs()), 'max h_feats_rec before layers and noise ')

        # random noise:
        if self.noise_initial > 0:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * torch.randn_like(h_feats_rec)
            coords_lig = coords_lig + noise_level * torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * torch.randn_like(coords_rec)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'h_feats_lig before layers but after noise ')
            log(torch.max(h_feats_rec.abs()), 'h_feats_rec before layers but after noise ')

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])],
                                    dim=1)
            h_feats_rec = torch.cat(
                [h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), torch.norm(h_feats_lig),
                'max and norm of h_feats_lig before layers but after noise and mu_r_norm')
            log(torch.max(h_feats_rec.abs()), torch.norm(h_feats_lig),
                'max and norm of h_feats_rec before layers but after noise and mu_r_norm')

        original_ligand_node_features = h_feats_lig
        original_receptor_node_features = h_feats_rec
        lig_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec_graph.edata['feat'] *= self.use_edge_features_in_gmn

        mask = None
        if self.cross_msgs:
            mask = get_mask(lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(), self.device)
        if self.separate_lig:
            coords_lig_separate =coords_lig
            h_feats_lig_separate =h_feats_lig
            coords_rec_separate =coords_rec
            h_feats_rec_separate =h_feats_rec
        full_trajectory = [coords_lig.detach().cpu()]
        geom_losses = 0
        # rotations = []
        # translations = []
        for i, layer in enumerate(self.iset_layers):
            if self.debug: log('layer ', i)
            coords_lig, \
            h_feats_lig, \
            coords_rec, \
            h_feats_rec, trajectory, geom_loss = layer(lig_graph=lig_graph,
                                rec_graph=rec_graph,
                                coords_lig=coords_lig,
                                h_feats_lig=h_feats_lig,
                                original_ligand_node_features=original_ligand_node_features,
                                orig_coords_lig=orig_coords_lig,
                                coords_rec=coords_rec,
                                h_feats_rec=h_feats_rec,
                                original_receptor_node_features=original_receptor_node_features,
                                orig_coords_rec=orig_coords_rec,
                                mask=mask,
                                geometry_graph=geometry_graph
                                )
            if not self.separate_lig:
                geom_losses = geom_losses + geom_loss
                full_trajectory.extend(trajectory)
                # rotations.append(rotation)
                # translations.append(translation)
        if self.separate_lig:
            for i, layer in enumerate(self.iset_layers_separate):
                if self.debug: log('layer ', i)
                coords_lig_separate, \
                h_feats_lig_separate, \
                coords_rec_separate, \
                h_feats_rec_separate, trajectory, geom_loss = layer(lig_graph=lig_graph,
                                    rec_graph=rec_graph,
                                    coords_lig=coords_lig_separate,
                                    h_feats_lig=h_feats_lig_separate,
                                    original_ligand_node_features=original_ligand_node_features,
                                    orig_coords_lig=orig_coords_lig,
                                    coords_rec=coords_rec_separate,
                                    h_feats_rec=h_feats_rec_separate,
                                    original_receptor_node_features=original_receptor_node_features,
                                    orig_coords_rec=orig_coords_rec,
                                    mask=mask,
                                    geometry_graph=geometry_graph
                                    )
                geom_losses = geom_losses + geom_loss
                full_trajectory.extend(trajectory)
                # rotations.append(rotation)
                # translations.append(translation)
        if self.save_trajectories:
            save_name = '_'.join(complex_names)
            torch.save({'trajectories': full_trajectory, 'names': complex_names}, f'data/results/trajectories/{save_name}.pt')
        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'max h_feats_lig after MPNN')
            log(torch.max(coords_lig.abs()), 'max coords_lig before after MPNN')

        rotations = []
        translations = []
        base_loss = 0
        vector_loss = []

        ligs_evolved = []
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx.insert(0, 0)

        if self.evolve_only:
            for idx in range(len(ligs_node_idx) - 1):
                lig_start = ligs_node_idx[idx]
                lig_end = ligs_node_idx[idx + 1]
                Z_lig_coords = coords_lig[lig_start:lig_end]
                ligs_evolved.append(Z_lig_coords)
            return [rotations, translations, ligs_evolved, geom_losses]

        max_scale_radius = (torch.exp(torch.tensor(1)) - torch.tensor(1)).to(self.device)

        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            rec_start = recs_node_idx[idx]
            rec_end = recs_node_idx[idx + 1]

            node_upd_per_ligand = h_feats_lig[lig_start:lig_end]
            node_upd_per_receptor = h_feats_rec[rec_start:rec_end]
            x_evolved_per_lig = coords_lig[lig_start:lig_end]
            x_evolved_per_rec = coords_rec[rec_start:rec_end]

            if self.separate_lig:
                ligs_evolved.append(coords_lig_separate[lig_start:lig_end])
                continue

            d = node_upd_per_ligand.shape[1]

            lig_feats_mean = torch.mean(self.h_mean_lig(node_upd_per_ligand), dim=0, keepdim=True)  # (1, d)
            rec_feats_mean = torch.mean(self.h_mean_rec(node_upd_per_receptor), dim=0, keepdim=True)  # (1, d)
            lig_centroid = torch.mean(x_evolved_per_lig, dim=0, keepdim=True)
            rec_centroid = torch.mean(x_evolved_per_rec, dim=0, keepdim=True)
            # lig_rec_distance = torch.linalg.norm(lig_centroid - rec_centroid)
            lig_max_radius = max_scale_radius * torch.max(torch.linalg.vector_norm(x_evolved_per_lig - lig_centroid, dim=1))
            rec_max_radius = max_scale_radius * torch.max(torch.linalg.vector_norm(x_evolved_per_rec - rec_centroid, dim=1))

            # attn_lig_fixed = (self.keypts_attention_lig_fixed(node_upd_per_ligand).view(-1, 1, d).transpose(0, 1) @
            #                self.keypts_queries_lig_fixed(rec_feats_mean).view(1, 1, d).transpose(0,1).transpose(1, 2) /
            #                math.sqrt(d)).view(1, -1)

            # if not self.lig_no_softmax:
            #     attn_lig_fixed = torch.softmax(attn_lig_fixed, dim=1)
            
            # lig_fixed_vec = self.feat_2_coord(attn_lig_fixed @ node_upd_per_ligand)

            attn_lig_rot = (self.keypts_attention_lig_rot(node_upd_per_ligand).view(-1, self.num_att_heads, d).transpose(0, 1) @
                            self.keypts_queries_lig_rot(rec_feats_mean).view(1, self.num_att_heads, d).transpose(0,1).transpose(1, 2) /
                            math.sqrt(d)).view(self.num_att_heads, -1)

            if not self.lig_no_softmax:
                attn_lig_rot = torch.exp(torch.softmax(attn_lig_rot, dim=1))
            # else:
            #     attn_lig_rot = torch.tanh(attn_lig_rot)

            attn_rec_rot = (self.keypts_attention_rec_rot(node_upd_per_receptor).view(-1, self.num_att_heads, d).transpose(0, 1) @
                            self.keypts_queries_rec_rot(lig_feats_mean).view(1, self.num_att_heads, d).transpose(0,1).transpose(1, 2) /
                            math.sqrt(d)).view(self.num_att_heads, -1)

            if not self.rec_no_softmax:
                attn_rec_rot = torch.exp(torch.softmax(attn_rec_rot, dim=1))
            # else:
            #     attn_rec_rot = torch.tanh(attn_rec_rot)

            lig_rot_vec = attn_lig_rot @ x_evolved_per_lig - lig_centroid
            # lig_rot_vec = (lig_rot_vec.T / torch.linalg.vector_norm(lig_rot_vec, dim=1)).T
            rec_rot_vec = rec_centroid - attn_rec_rot @ x_evolved_per_rec
            # rec_rot_vec = (rec_rot_vec.T / torch.linalg.vector_norm(rec_rot_vec, dim=1)).T

            # Ver 1
            # rotation_matrix = self.W_feature(node_upd_per_receptor) @ node_upd_per_ligand.T + \
            #                 (self.W_feature(node_upd_per_ligand) @ node_upd_per_receptor.T).T

            # rotation_matrix = self.R_linear(x_evolved_per_rec.T @ rotation_matrix @ x_evolved_per_lig)
            # rotation_matrix = apply_norm(None, rotation_matrix, "LN", self.R_norm)

            if self.locknkey == "direct":
                # Ver 2: Directly
                rotation_matrix, b_l = get_rotation_matrix(lig_rot_vec[0], rec_rot_vec[0], device=self.device)
                base_loss += b_l
                # vector_loss_cplx = get_vector_loss(lig_rot_vec, mean=lig_max_radius) + get_vector_loss(rec_rot_vec, mean=rec_max_radius) 

            elif self.locknkey == "direct_base":
                # rotation_matrix = rec_rot_vec.T @ torch.linalg.inv(lig_rot_vec.T)
                # base_loss += get_base_loss(rec_rot_vec) + get_base_loss(lig_rot_vec)

                reference_vector = normalize(lig_rot_vec[1] + rec_rot_vec[1])
                rotation_matrix, b_l = get_rotation_matrix(lig_rot_vec[0], rec_rot_vec[0], reference_vector=reference_vector, device=self.device)
                base_loss += b_l
                vector_loss_cplx = get_vector_loss(lig_rot_vec, mean=lig_max_radius) + get_vector_loss(rec_rot_vec, mean=rec_max_radius) 

                # rotation_matrix, b_l = get_rotation_matrix(lig_rot_vec[0], rec_rot_vec[0], reference_vector=[lig_rot_vec[1], rec_rot_vec[1]], device=self.device)
                # base_loss += b_l

            elif self.locknkey == "indirect":
                # Ver 3: Indirectly
                lig_rot_vec_xy = lig_rot_vec * self.xy_mask
                lig_rot_vec_yz = lig_rot_vec * self.yz_mask
                lig_rot_vec_zx = lig_rot_vec * self.zx_mask

                rec_rot_vec_xy = rec_rot_vec * self.xy_mask
                rec_rot_vec_yz = rec_rot_vec * self.yz_mask
                rec_rot_vec_zx = rec_rot_vec * self.zx_mask

                # v1: [x1; y1]: lig_fixed
                # v2: [x2; y2]: rec_rot
                # atan2(x1*y2 - y1*x2; x1*x2 + y1*y2)
                
                alpha = torch.atan2(lig_rot_vec_xy[:,0] * rec_rot_vec_xy[:,1] - lig_rot_vec_xy[:,1] * rec_rot_vec_xy[:,0],
                                    lig_rot_vec_xy[:,0] * rec_rot_vec_xy[:,0] + lig_rot_vec_xy[:,1] * rec_rot_vec_xy[:,1])

                beta = torch.atan2(lig_rot_vec_zx[:,0] * rec_rot_vec_zx[:,1] - lig_rot_vec_zx[:,1] * rec_rot_vec_zx[:,0],
                                    lig_rot_vec_zx[:,0] * rec_rot_vec_zx[:,0] + lig_rot_vec_zx[:,1] * rec_rot_vec_zx[:,1])

                gamma = torch.atan2(lig_rot_vec_yz[:,0] * rec_rot_vec_yz[:,1] - lig_rot_vec_yz[:,1] * rec_rot_vec_yz[:,0],
                                    lig_rot_vec_yz[:,0] * rec_rot_vec_yz[:,0] + lig_rot_vec_yz[:,1] * rec_rot_vec_yz[:,1])

                R_z = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).to(self.device)
                R_y = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(self.device)
                R_x = torch.FloatTensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).to(self.device)

                R_z[0][0] = torch.cos(alpha)
                R_z[0][1] = -torch.sin(alpha)
                R_z[1][0] = torch.sin(alpha)
                R_z[1][1] = torch.cos(alpha)

                R_y[0][0] = torch.cos(beta)
                R_y[0][2] = torch.sin(beta)
                R_y[2][0] = -torch.sin(beta)
                R_y[2][2] = torch.cos(beta)

                R_x[1][1] = torch.cos(gamma)
                R_x[1][2] = -torch.sin(gamma)
                R_x[2][1] = torch.sin(gamma)
                R_x[2][2] = torch.cos(gamma)

                rotation_matrix = R_z @ R_y @ R_x
                vector_loss_cplx = get_vector_loss(lig_rot_vec[0]) + get_vector_loss(rec_rot_vec[0])

            else:
                rotation_matrix = torch.eye(3).to(self.device)
                vector_loss_cplx = 0
                
            x_evolved_per_lig = (rotation_matrix @ (x_evolved_per_lig).T).T
            # x_evolved_per_lig = (rotation_matrix @ (x_evolved_per_lig-lig_centroid).T).T + lig_centroid

            att_weights_rec_trans = (self.keypts_attention_rec_trans(node_upd_per_receptor).view(-1, 1, d).transpose(0, 1) @
                            self.keypts_queries_rec_trans(lig_feats_mean).view(1, 1, d).transpose(0,1).transpose(1, 2) /
                            math.sqrt(d)).view(1, -1)

            if not self.rec_no_softmax:
                att_weights_rec_trans = torch.softmax(att_weights_rec_trans, dim=1)

            predicted_lig_centroid = att_weights_rec_trans @ x_evolved_per_rec
            translation_vector = predicted_lig_centroid - torch.mean(x_evolved_per_lig, dim=0, keepdim=True)

            if self.debug:
                log('rotation', rotation_matrix)
                log('rotation @ rotation.t() - eye(3)', rotation_matrix @ rotation_matrix.T - torch.eye(3).to(self.device))
                log('translation', translation_vector)
                log(torch.max(x_evolved_per_lig.abs()), 'x_i new')
                log("Centroid changed?", lig_centroid - torch.mean(x_evolved_per_lig, dim=0, keepdim=True))

            ligs_evolved.append(x_evolved_per_lig + translation_vector)
            vector_loss.append(vector_loss_cplx)

            rotations.append(rotation_matrix)
            translations.append(translation_vector)

        return [rotations, translations, ligs_evolved, geom_losses, base_loss, vector_loss]

    def __repr__(self):
        return "ISET " + str(self.__dict__)

# =================================================================================================================


class SeDok(nn.Module):

    def __init__(self, device='cuda:0', debug=False, use_evolved_lig=False, evolve_only=False, **kwargs):
        super(SeDok, self).__init__()
        self.debug = debug
        self.evolve_only = evolve_only
        self.use_evolved_lig = use_evolved_lig
        self.device = device
        self.iset = ISET(device=self.device, debug=self.debug, evolve_only=self.evolve_only, **kwargs)
        if self.debug:
            torch.autograd.set_detect_anomaly(True)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, lig_graph, rec_graph, geometry_graph=None, complex_names=None, epoch=0):
        if self.debug: log(complex_names)
        predicted_ligs_coords_list = []
        outputs = self.iset(lig_graph, rec_graph, geometry_graph, complex_names, epoch)
        evolved_ligs = outputs[2]
        if self.evolve_only:
            return evolved_ligs, None, None, outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        for idx in range(len(ligs_node_idx) - 1):
            start = ligs_node_idx[idx]
            end = ligs_node_idx[idx + 1]
            orig_coords_lig = lig_graph.ndata['new_x'][start:end]

            if self.use_evolved_lig:
                predicted_coords = evolved_ligs[idx]
            else:
                predicted_coords = orig_coords_lig

            if self.debug:
                log('\n ---> predicted_coords mean - true ligand mean ',
                    predicted_coords.mean(dim=0) - lig_graph.ndata['x'].mean(dim=0), '\n')
            predicted_ligs_coords_list.append(predicted_coords)
        #torch.save({'predictions': predicted_ligs_coords_list, 'names': complex_names})
        return predicted_ligs_coords_list, None, None, outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]

    def __repr__(self):
        return "SeDok " + str(self.__dict__)
