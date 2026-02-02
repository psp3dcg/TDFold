'''All atom potential energy loss function'''
import torch
import pickle as pkl
import networkx as nx
import numpy as np
import torch_geometric as tg
from torch_geometric.utils import degree


def bond_potential_energy_loss(new_out_coor, target_bond_length_value, edge_index, k = 2.0):

    '''
    Compute the bond potential energy of atoms

    Input:
        - new_out_coor(tensor): all atom coordiantes
        - target_bond_length_value(tensor): true bond length of protein
        - edge_index(tensor): the indices of source node and destination node in edges(bonds) for atom graph
        - k(float): bond energy hyper-parameter
    Output:
        - bond_energy(tensor): the bond energy of protein
    '''
    def euclidean_distances(a, b):
        assert a.shape == b.shape
        direct_vec = a - b

        dist = torch.norm(direct_vec, dim=-1)
        return dist

    predict_src_atom_coor = new_out_coor[edge_index[0]]
    predict_dst_atom_coor = new_out_coor[edge_index[1]]
    predict_bond_length = euclidean_distances(predict_src_atom_coor, predict_dst_atom_coor)


    bond_dist_diff = predict_bond_length - target_bond_length_value*10
    bond_dist_diff = torch.sqrt(0.5 * torch.sum(bond_dist_diff * bond_dist_diff)+1e-8)
    bond_energy = 0.5 * k * bond_dist_diff

    return bond_energy

def angle_potential_energy_loss(new_out_coor, target_angle_value, angle_index, k = 2.0):

    '''
    Compute the angle potential energy of atoms

    Input:
        - new_out_coor(tensor): all atom coordiantes
        - target_angle_value(tensor): true angle value of protein
        - angle_index(tensor): the indices of nodes forming angles in proteins
        - k(float): angle energy hyper-parameter
    Output:
        - angle_energy(tensor): the angle energy of protein
    '''
    def get_angle_vectors(atom_coor, angle_atom_index):

        '''
        Takes atom coordinates and angle_atom_index and output two vectors forming the angle

        Input:
            - atom_coor(tensor): all atom coordinates
            - angle_atom_index(tensor): the indices of nodes forming angles in proteins
        Output:
            - result(tensor): the cosine value of two batches of vectors
        '''
        direct_vec_1 = atom_coor[angle_atom_index[:,1]] - atom_coor[angle_atom_index[:,0]]
        direct_vec_2 = atom_coor[angle_atom_index[:,1]] - atom_coor[angle_atom_index[:,2]]
        result = torch.cat([direct_vec_1, direct_vec_2], dim=1).reshape(-1, 2, 3)
        
        return result

    def batch_cos_between_vectors(a, b):

        '''
        Compute cosine value between two batches of input vectors

        Input:
            - a(tensor): first batch of input vectors
            - b(tensor): second batch of input vectors
        Output:
            - cos(tensor): the cosine value of two batches of vectors
        '''

        inner_product = (a * b).sum(dim=-1)

        # norms
        a_norm = torch.linalg.norm(a, dim=-1)
        b_norm = torch.linalg.norm(b, dim=-1)

        # protect denominator during division
        den = a_norm * b_norm + 1e-6
        cos = inner_product / den

        return cos

    right_index = torch.where(target_angle_value >= 0)[0]
    angle_index = angle_index[right_index]
    target_angle_value = target_angle_value[right_index]
    predict_direct_vec_set = get_angle_vectors(new_out_coor, angle_index)
    predict_cos_value = batch_cos_between_vectors(predict_direct_vec_set[:,0,:], 
                                                    predict_direct_vec_set[:,1,:])

    predict_angle_value = torch.acos(predict_cos_value)
    bond_angle_diff = predict_angle_value - target_angle_value
    bond_angle_diff = torch.sqrt(torch.sum(bond_angle_diff * bond_angle_diff)+1e-8)
    angle_energy = 0.5 * k * bond_angle_diff

    return angle_energy



def dihedral_potential_energy_loss(new_out_coor, target_dihedral_value, dihedral_index):

    '''
    Compute the dihedral(torsion) potential energy of atoms

    Input:
        - new_out_coor(tensor)
        - target_dihedral_value(tensor)
        - dihedral_index(tensor)
    Output:
        - dihedral_energy(tensor): the cosine value of two batches of vectors
    '''

    def batch_dihedrals(p0, p1, p2, p3, angle=False):

        '''
        Compute cosine value between two batches of input vectors

        Input:
            - p0(tensor): the coordinate of particle 1
            - p1(tensor): the coordinate of particle 2
            - p2(tensor): the coordinate of particle 3 
            - p3(tensor): the coordinate of particle 4
            - angle(bool): return degrees or radians
        Output:
            - angle_value(tensor): degree of angle
            - sin_d_/den(tensor): sine value of angle
            - cos_d_/den(tensor): cosine value of angle
        '''
        s1 = p1 - p0
        s2 = p2 - p1
        s3 = p3 - p2
        sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
        cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        if angle:
            angle_value = torch.abs(torch.atan2(sin_d_, cos_d_ + 1e-10))
            return angle_value
        else:
            return sin_d_/den, cos_d_/den

        

    def get_dihedral_atom_positions(dihedral_atom_index_set, atom_position):
        '''
        Input:
            - dihedral_atom_index_set(tensor):two atoms on dihedral rotation axis N * 2
            - atom_position(tensor):all atom positions
        Output:
            - dihedral_atom_position_set(tensor):the position of four atoms building the dihedral M * 4
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

    n_per_set = target_dihedral_value[:, 0, :] #periodicity
    phase_set = target_dihedral_value[:, 1, :]
    k_set = target_dihedral_value[:, 2, :]
    pred_dihedral_torsion_set *= 0.5

    dihedral_energy = 0
    for i in range(k_set.shape[-1]):

        dihedral_diff = k_set[:,i]*(1+torch.cos(n_per_set[:,i]*pred_dihedral_torsion_set - phase_set[:,i]))
        dihedral_energy += 0.5*torch.sum(dihedral_diff)

    return dihedral_energy
