import torch
import pickle as pkl
import networkx as nx
import numpy as np
import torch_geometric as tg
from torch_geometric.utils import degree




residue_index_dict = {
        0:"ALA",
        1:"CYS",
        2:"ASP",
        3:"GLU",
        4:"PHE",
        5:"GLY",
        6:"HIS",
        7:"ILE",
        8:"LYS",
        9:"LEU",
        10:"MET",
        11:"ASN",
        12:"PRO",
        13:"GLN",
        14:"ARG",
        15:"SER",
        16:"THR",
        17:"VAL",
        18:"TRP",
        19:"TYR"
    }

def read_force_gt_pkl():
    with open('atom_type.pkl', 'rb') as f:
        atom_type_dict = pkl.load(f)
    with open('bond_force.pkl', 'rb') as f:
        bond_force_dict = pkl.load(f)
    with open('angle_force.pkl', 'rb') as f:
        angle_force_dict = pkl.load(f)
    with open('dihedral_force.pkl', 'rb') as f:
        dihedral_force_dict = pkl.load(f)

    return atom_type_dict, bond_force_dict, angle_force_dict, dihedral_force_dict

def get_atom_type_list(atom_feature):
    def one_hot_to_number(one_hot_vector):
        row_idx, col_idx = torch.where(one_hot_vector==1)
        return col_idx
    atom_res_type_one_hot = atom_feature[:,4:24]
    atom_res_type = one_hot_to_number(atom_res_type_one_hot)

    



def cross_loss_mask(pred_, true_, mask, device):
    '''pair distance and angles cross entropy loss

    Input:
        - pred_(tensor):predict distance, omega, theta and phi
        - true_(tensor):true distance, omega, theta and phi
        - mask(tensor):distance, omega, theta and phi mask
        - device(str):cpu or cuda
    Output:
        - result(tensor):loss value
    '''
    pred_ = pred_.reshape(-1, pred_.shape[-1])
    true_ = torch.flatten(true_).to(device)
    mask = torch.flatten(mask).float().to(device)
    cross_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss = cross_func(pred_, true_)
    loss = mask * loss
    result = torch.mean(loss)
    return result

def kl_divergence(prob):
    """Compute the KL divergence between two discrete probability distributions
        The calculation is done directly using the Kullback-Leibler divergence,
        """
    # pdb.set_trace()
    # prob +- offset prevent 0 or 1
    if torch.max(prob) == 1:
        prob = prob - 1e-4
    if torch.min(prob) == 0:
        prob = prob + 1e-4
    expect_prob=torch.ones_like(prob)*0.005 # 0.5
    kl_div=prob*torch.log(prob/expect_prob)+(1-prob)*torch.log((1-prob)/(1-expect_prob))
    kl_loss=torch.mean(kl_div)
    return kl_loss

def harmonic_bond_force_loss(new_out_coor, target_bond_length_value, edge_index):

    def euclidean_distances(a, b):
        assert a.shape == b.shape
        direct_vec = a - b

        dist = torch.norm(direct_vec, dim=-1)
        return dist


    # HarmonicBondForce
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    predict_src_atom_coor = new_out_coor[edge_index[0]]
    predict_dst_atom_coor = new_out_coor[edge_index[1]]

    # pdb.set_trace()
    predict_bond_length = euclidean_distances(predict_src_atom_coor, predict_dst_atom_coor)

    # target_src_atom_coor = target_coor[edge_index[0]]
    # target_dst_atom_coor = target_coor[edge_index[1]]
    # target_bond_dist = EuclideanDistances(target_src_atom_coor,target_dst_atom_coor)

    bond_dist_diff = predict_bond_length - target_bond_length_value*10

    # each bond has been computed twice, so devided by 2


    print('bond_diff_max',torch.max(bond_dist_diff * bond_dist_diff).item())
    print('bond_diff_mean',torch.mean(bond_dist_diff * bond_dist_diff).item())
    print('bond_diff_min',torch.min(bond_dist_diff * bond_dist_diff).item())

    # bond_dist_diff = torch.sqrt(loss_fn(predict_bond_length, target_bond_length_value*10)+1e-8)

    k = 2.0

    bond_dist_diff = torch.sqrt(0.5 * torch.sum(bond_dist_diff * bond_dist_diff)+1e-8)

    e_bond_force = 0.5 * k * bond_dist_diff

    return e_bond_force

def harmonic_angle_force_loss(new_out_coor, target_angle_value, angle_index):
    def get_neighbor_bonds(atom_coor, angle_atom_index):
        """
        Takes the edge indices and bond type and returns dictionary mapping atom index to neighbor bond types
        Note: this only includes atoms with degree > 1
        """
        # start, end = edge_index
        # idxs, vals = torch.unique(start, return_counts=True)
        # new_idxs = idxs[torch.where(vals > 1)]

        
        # result = []
        # for i in new_idxs:
        #     center_atom_index = torch.where(edge_index[0] == i)
        #     edge_src_index = edge_index[0][center_atom_index]
        #     edge_dst_index = edge_index[1][center_atom_index]

        #     direct_vecs = atom_coor[edge_src_index] - atom_coor[edge_dst_index]
        #     direct_vecs_index = torch.tensor([i for i in range(direct_vecs.shape[0])], dtype=torch.long)
        #     # pdb.set_trace()
        #     vec_idx_combinations = torch.combinations(direct_vecs_index)



        #     result.append(direct_vecs[vec_idx_combinations])

        # result = torch.cat(result, dim=0)

        direct_vec_1 = atom_coor[angle_atom_index[:,1]] - atom_coor[angle_atom_index[:,0]]
        direct_vec_2 = atom_coor[angle_atom_index[:,1]] - atom_coor[angle_atom_index[:,2]]

        result = torch.cat([direct_vec_1, direct_vec_2], dim=1).reshape(-1, 2, 3)

        
        return result

    def batch_cos_between_vectors(a, b):
        """
        Compute cosine value between two batches of input vectors
        """
        # pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        inner_product = (a * b).sum(dim=-1)

        # norms
        a_norm = torch.linalg.norm(a, dim=-1)
        b_norm = torch.linalg.norm(b, dim=-1)

        # protect denominator during division
        den = a_norm * b_norm + 1e-6
        cos = inner_product / den

        return cos
    # import pdb
    # pdb.set_trace()
    #loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    right_index = torch.where(target_angle_value >= 0)[0]
    angle_index = angle_index[right_index]
    target_angle_value = target_angle_value[right_index]
    predict_direct_vec_set = get_neighbor_bonds(new_out_coor, angle_index)
    predict_cos_value = batch_cos_between_vectors(predict_direct_vec_set[:,0,:], 
                                                    predict_direct_vec_set[:,1,:])


    predict_angle_value = torch.acos(predict_cos_value)
    # print(torch.where(predict_cos_value > 1))
    # print(torch.where(predict_cos_value < -1))
    # import pdb
    # pdb.set_trace()
    
    bond_angle_diff = predict_angle_value - target_angle_value
    # print('raw_angle_diff', bond_angle_diff.item())
    # print('angle_diff^2', bond_angle_diff.item())
    print('angle_diff_max',torch.max(bond_angle_diff * bond_angle_diff).item())
    print('angle_diff_mean',torch.mean(bond_angle_diff * bond_angle_diff).item())
    print('angle_diff_min',torch.min(bond_angle_diff * bond_angle_diff).item())
    bond_angle_diff = torch.sqrt(torch.sum(bond_angle_diff * bond_angle_diff)+1e-8)
    #bond_angle_diff = torch.sqrt(loss_fn(predict_angle_value, target_angle_value)+1e-8)


    k = 2.0
    e_angle_force = 0.5 * k * bond_angle_diff

    
    return e_angle_force


def get_neighbors(edge_index):

    start, end = edge_index
    degrees = degree(end)
    dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
    dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)
    # # first method which removes one (pseudo) random edge from a cycle
    dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

    # prioritize rings for assigning dihedrals
    dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]

    start, end = edge_index
    start_idxs, start_vals = torch.unique(start, return_counts=True)
    end_idxs, end_vals = torch.unique(end, return_counts=True)
    print("edge_index", edge_index[:, -20:])
    print("start_edge_index_idxs", start_idxs)
    print("start_edge_index_vals", start_vals)
    print("end_edge_index_idxs", end_idxs)
    print("end_edge_index_vals", end_vals)
    print("dihedral_pairs_true", dihedral_pairs)
    new_idxs = start_idxs[torch.where(start_vals > 1)]

    
    neighbor_result = {}
    for i in new_idxs:
        center_atom_index = torch.where(edge_index[0] == i)
        edge_dst_index = edge_index[1][center_atom_index]

        neighbor_result[i.item()] = edge_dst_index

    return neighbor_result

def periodic_torsion_force_loss(new_out_coor, target_dihedral_value, dihedral_index):
    
    def batch_dihedrals(p0, p1, p2, p3, angle=False):
        # import pdb
        # pdb.set_trace()

        s1 = p1 - p0
        s2 = p2 - p1
        s3 = p3 - p2

        
        sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
        cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        if angle:
            return torch.abs(torch.atan2(sin_d_, cos_d_ + 1e-10))
            # return torch.acos(cos_d_ / den)

        else:
            # den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
            return sin_d_/den, cos_d_/den

        # n1, n2 = torch.cross(s1, s2), torch.cross(s2, s3)

        # dihedral_torsion = torch.acos()

        

    def get_dihedral_atom_positions(dihedral_atom_index_set, atom_position):
        '''
        Input:
            - dihedral_pairs(tensor):two atoms on dihedral rotation axis N * 2
            - atom_position(tensor):all atom positions
        Output:
            - dihedral_atom_positions(tensor):the position of four atoms building the dihedral M * 4
        '''
        dihedral_atom_position_set = []
        for atom_index in dihedral_atom_index_set:
            dihedral_atom_position_set.append(atom_position[atom_index])


        dihedral_atom_position_set = torch.cat(dihedral_atom_position_set).view(-1,4,3)

        return dihedral_atom_position_set


    # function main part

    true_index = torch.where(target_dihedral_value>=0)[0]
    target_dihedral_value = target_dihedral_value[true_index]
    dihedral_index = dihedral_index[true_index]


    pred_dihedral_atom_position_set = get_dihedral_atom_positions(dihedral_index, 
                                                                  new_out_coor)


    pred_dihedral_torsion_set = batch_dihedrals(pred_dihedral_atom_position_set[:,0,:],
                                                pred_dihedral_atom_position_set[:,1,:],
                                                pred_dihedral_atom_position_set[:,2,:],
                                                pred_dihedral_atom_position_set[:,3,:],
                                                True)


    # import pdb
    # pdb.set_trace()    
    n_per_set = target_dihedral_value[:, 0, :] #periodicity
    phase_set = target_dihedral_value[:, 1, :]
    k_set = target_dihedral_value[:, 2, :]
    # import pdb
    # pdb.set_trace()

    pred_dihedral_torsion_set *= 0.5

    e_dihedral_force = 0

    for i in range(k_set.shape[-1]):

        dihedral_diff = k_set[:,i]*(1+torch.cos(n_per_set[:,i]*pred_dihedral_torsion_set - phase_set[:,i]))
        print('dihedral_diff_max',torch.max(dihedral_diff).item())
        print('dihedral_diff_mean',torch.mean(dihedral_diff).item())
        print('dihedral_diff_min',torch.min(dihedral_diff).item())
        e_dihedral_force += 0.5*torch.sum(dihedral_diff)

    return e_dihedral_force

def periodic_torsion_force_loss_2(new_out_coor, bb_true_coor, dihedral_index):
    
    def batch_dihedrals(p0, p1, p2, p3, angle=False):
        # import pdb
        # pdb.set_trace()

        s1 = p1 - p0
        s2 = p2 - p1
        s3 = p3 - p2

        
        sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
        cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        if angle:
            return torch.abs(torch.atan2(sin_d_, cos_d_ + 1e-10))
            # return torch.acos(cos_d_ / den)

        else:
            # den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
            return sin_d_/den, cos_d_/den

        # n1, n2 = torch.cross(s1, s2), torch.cross(s2, s3)

        # dihedral_torsion = torch.acos()

        

    def get_dihedral_atom_positions(dihedral_atom_index_set, atom_position):
        '''
        Input:
            - dihedral_pairs(tensor):two atoms on dihedral rotation axis N * 2
            - atom_position(tensor):all atom positions
        Output:
            - dihedral_atom_positions(tensor):the position of four atoms building the dihedral M * 4
        '''

        # import pdb 
        # pdb.set_trace()
        dihedral_atom_position_set = []
        for atom_index in dihedral_atom_index_set:
            dihedral_atom_position_set.append(atom_position[atom_index])


        dihedral_atom_position_set = torch.cat(dihedral_atom_position_set).view(-1,4,3)

        return dihedral_atom_position_set


    # function main part

    # true_index = torch.where(target_dihedral_value>=0)[0]
    # target_dihedral_value = target_dihedral_value[true_index]
    # dihedral_index = dihedral_index[true_index]


    pred_dihedral_atom_position_set = get_dihedral_atom_positions(dihedral_index, 
                                                                  new_out_coor)


    pred_dihedral_torsion_set = batch_dihedrals(pred_dihedral_atom_position_set[:,0,:],
                                                pred_dihedral_atom_position_set[:,1,:],
                                                pred_dihedral_atom_position_set[:,2,:],
                                                pred_dihedral_atom_position_set[:,3,:],
                                                True)

    true_dihedral_atom_position_set = get_dihedral_atom_positions(dihedral_index, 
                                                                  bb_true_coor)


    true_dihedral_torsion_set = batch_dihedrals(true_dihedral_atom_position_set[:,0,:],
                                                true_dihedral_atom_position_set[:,1,:],
                                                true_dihedral_atom_position_set[:,2,:],
                                                true_dihedral_atom_position_set[:,3,:],
                                                True)

    dihedral_diff = pred_dihedral_torsion_set - true_dihedral_torsion_set

    dihedral_diff = 0.5 * torch.pow(dihedral_diff, 2)
    print('dihedral_diff_max',torch.max(dihedral_diff).item())
    print('dihedral_diff_mean',torch.mean(dihedral_diff).item())
    print('dihedral_diff_min',torch.min(dihedral_diff).item())

    e_dihedral_force = 0.5 * torch.sqrt(torch.sum(dihedral_diff)+1e-8)



    # import pdb
    # pdb.set_trace()    
    # n_per_set = target_dihedral_value[:, 0, :] #periodicity
    # phase_set = target_dihedral_value[:, 1, :]
    # k_set = target_dihedral_value[:, 2, :]
    # # import pdb
    # # pdb.set_trace()

    # pred_dihedral_torsion_set *= 0.5

    # e_dihedral_force = 0

    # for i in range(k_set.shape[-1]):

    #     dihedral_diff = k_set[:,i]*(1+torch.cos(n_per_set[:,i]*pred_dihedral_torsion_set - phase_set[:,i]))
    #     print('dihedral_diff_max',torch.max(dihedral_diff).item())
    #     print('dihedral_diff_mean',torch.mean(dihedral_diff).item())
    #     print('dihedral_diff_min',torch.min(dihedral_diff).item())
    #     e_dihedral_force += 0.5*torch.sum(dihedral_diff)



    return e_dihedral_force

