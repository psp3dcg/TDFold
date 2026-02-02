import os
import torch
import itertools
import numpy as np
import pickle as pkl
import networkx as nx
import numpy.linalg as lin
import torch_geometric as tg
from torch_geometric.utils import degree
from torch_geometric.data import Data
from Bio.PDB.PDBParser import PDBParser
from all_atom_fold.combine_bb_sc import add_side_chain_atom

# read the atom types, bond lengths, angles, and dihedral parameters defined in OpenMM
atom_type_info_path = "all_atom_fold/force_file/atom_type.pkl"
bond_force_info_path = "all_atom_fold/force_file/bond_force.pkl"
angle_force_info_path = "all_atom_fold/force_file/angle_force.pkl"
dihedral_force_info_path = "all_atom_fold/force_file/dihedral_force.pkl"
with open(atom_type_info_path, 'rb') as f:
    atom_type_dict = pkl.load(f)

with open(bond_force_info_path, 'rb') as f:
    bond_force_dict = pkl.load(f)

with open(angle_force_info_path, 'rb') as f:
    angle_force_dict = pkl.load(f)

with open(dihedral_force_info_path, 'rb') as f:
    dihedral_force_dict = pkl.load(f)


def EuclideanDistances(a, b):
    '''
    Compute euclidean distance between two point sets with same shape

    Input:
        - a(tensor(N, 3)): the coordinate of point set a
        - b(tensor(N, 3)): the coordinate of point set b
    Output:
        - dist(tensor): the euclidean distance sets corresponding to point set a and b
    '''
    assert a.shape == b.shape
    direct_vec = a - b

    dist = torch.norm(direct_vec, dim=-1)
    return dist

def get_neighbors(edge_index):

    '''
    Get each node's adjacent node in graph
    
    Input:
        - edge_index(tensor(2, M)): the indices of source node and destination node in edges
    Output:
        - neighbor_result(dict): the dict of node's adjacent nodes, 
          key is the node index and value is the list containing adjacent nodes' indices
    '''
    start, end = edge_index
    degrees = degree(start)
    degrees_end = degree(end)
    dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
    dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)
    # first method which removes one (pseudo) random edge from a cycle
    dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

    # prioritize rings for assigning dihedrals
    dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]

    start, end = edge_index
    start_idxs, start_vals = torch.unique(start, return_counts=True)
    end_idxs, end_vals = torch.unique(end, return_counts=True)
    
    new_idxs = start_idxs[torch.where(start_vals > 1)]
    
    
    neighbor_result = {}
    for i in new_idxs:
        center_atom_index = torch.where(edge_index[0] == i)
        edge_dst_index = edge_index[1][center_atom_index]

        neighbor_result[i.item()] = edge_dst_index

    return neighbor_result


def get_angle_atom_index(edge_index):

    '''
    Takes the edge indices and returns the index triplet of atoms forming angles
    Note: this only includes atoms with degree > 1

    Input:
        - edge_index(tensor(2, M)): the indices of source node and destination node in edges
    Output:
        - result(list(N, 3)): the index of the three atoms that make up an angle 
    '''
    start, end = edge_index
    idxs, vals = torch.unique(start, return_counts=True)
    new_idxs = idxs[torch.where(vals > 1)]

    
    result = []
    for i in new_idxs:
        center_atom_index = torch.where(edge_index[0] == i)
        edge_dst_index = edge_index[1][center_atom_index]
        edge_dst_index_comb = torch.combinations(edge_dst_index)
        for idx_comb in edge_dst_index_comb:
            result.append([idx_comb[0].item(), i.item(), idx_comb[1].item()])
    
    return result

def get_dihedral_pair_index(data, edge_index, openmm_atom_type_list):

    '''
    Takes the edge indices and returns the index triplet of atoms forming angles
    Note: this only includes atoms with degree > 1
    
    Input:
        - data(torch geometric Data object): atom graph data
        - edge_index(tensor(2, M)): the indices of source node and destination node in edges
        - openmm_atom_type_list(list): the format of atom type in openmm tools
    Output:
        - final_dihedral_atom_index_set(tensor(N, 4)):the index of the four atoms that make up a dihedral
    '''
    def get_cycle_values(cycle_list, start_at=None):

        '''
        Make a generator for cycle values
            Input:
                - cycle_list(list): list containing all indices forming a cycle
                - start_at(int): the start index
        '''
        start_at = 0 if start_at is None else cycle_list.index(start_at)
        while True:
            yield cycle_list[start_at]
            start_at = (start_at + 1) % len(cycle_list)


    def get_cycle_indices(cycle, start_idx):

        '''
        Get the edge indices form a cycle (eg. [0, 1], [1, 2], [2, 3], [3, 0])
        
        Input:
            - cycle(list): the list containing all indices forming a cycle
            - start_idx(int): the start index
        Output:
            - indices(list): the list consists of the edge indices
        '''
        cycle_it = get_cycle_values(cycle, start_idx)
        indices = []

        end = 9e99
        start = next(cycle_it)
        a = start
        while start != end:
            b = next(cycle_it)
            indices.append(torch.tensor([a, b]))
            a = b
            end = b

        return indices


    def get_current_cycle_indices(cycles, cycle_check, idx):

        '''
        Get indices of current cycle
        
        Input:
            - cycle(list): the list containing all indices forming a cycle
            - cycle_check(list): the list of bool values of idx in the cycle or not
            - idx(int): the index of one node in pair
        Output:
            - cycle_indices(list): the list consists of the edge indices
        '''
        c_idx = [i for i, c in enumerate(cycle_check) if c][0]
        current_cycle = cycles.pop(c_idx)
        current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
        cycle_indices = get_cycle_indices(current_cycle, current_idx)

        return cycle_indices


    def get_dihedral_pairs(edge_index, data):

        '''
        Given edge indices, return pairs of indices for dihedrals
        
        Input:
            - edge_index(tensor(2, M)): the indices of source node and destination node in edges
            - data(torch geometric Data object): atom graph data
        Output:
            - result(tensor): the tensor consists of the indices forming dihedrals
        '''
        start, end = edge_index
        degrees = degree(start)
        dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
        dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)

        # first method which removes one (pseudo) random edge from a cycle
        dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

        # prioritize rings for assigning dihedrals
        dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]
        G = nx.to_undirected(tg.utils.to_networkx(data))
        cycles = nx.cycle_basis(G)
        keep, sorted_keep = [], []

        if len(dihedral_pairs.shape) == 1:
            dihedral_pairs = dihedral_pairs.unsqueeze(0)

        for pair in dihedral_pairs:
            x, y = pair

            if sorted(pair) in sorted_keep:
                continue

            y_cycle_check = [y in cycle for cycle in cycles]
            x_cycle_check = [x in cycle for cycle in cycles]

            if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle
                cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x)
                keep.extend(cycle_indices)

                sorted_keep.extend([sorted(c) for c in cycle_indices])
                continue

            if any(y_cycle_check):
                cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y)
                keep.append(pair)
                keep.extend(cycle_indices)

                sorted_keep.append(sorted(pair))
                sorted_keep.extend([sorted(c) for c in cycle_indices])
                continue

            keep.append(pair)
        keep = [t for t in keep]
        result = torch.stack(keep).t()

        return result

    def get_improper_dihedral_atom_index(edge_index, data, openmm_atom_type_list):

        '''
        Get the atom order of improper dihedral an improper torsion involves a central atom and 
        three others that are bonded to it: atoms 2, 3, and 4 are all bonded to atom 1.
        
        Input:
            - edge_index(tensor(2, M)): the indices of source node and destination node in edges
            - data(torch geometric Data object): atom graph data
            - openmm_atom_type_list(list): the format of atom type in openmm tools
        Output:
            - improper_dihedral_atom_index(tensor): the tensor consists of the indices forming improper dihedrals
        '''


        # relative mass of four heavy atoms
        mass_dict = {'C':12.0107,
                    'N':14.0067,
                    'O':15.9994,
                    'S':32.0655}
        num_of_atoms = data.x.shape[0]
        bondedToAtom = [set() for _ in range(num_of_atoms)]
        atomBonds = [set() for _ in range(num_of_atoms)]
        src_node_index = edge_index[0]
        dst_node_index = edge_index[1]

        improper_dihedral_atom_index_list = []

        for i in range(len(src_node_index)):
            bond_src_atom = src_node_index[i].item()
            bond_dst_atom = dst_node_index[i].item()
            bondedToAtom[bond_src_atom].add(bond_dst_atom)
            bondedToAtom[bond_dst_atom].add(bond_src_atom)
            atomBonds[bond_src_atom].add(i)
            atomBonds[bond_dst_atom].add(i)

        bondedToAtom = [sorted(b) for b in bondedToAtom]

        for atom_index in range(len(bondedToAtom)):
            bondedTo = bondedToAtom[atom_index]
            if len(bondedTo) > 2:
                for subset in itertools.combinations(bondedTo, 3):
                    atom_index_1 = subset[0]
                    atom_index_2 = subset[1]
                    element_1 = openmm_atom_type_list[atom_index_1]
                    element_2 = openmm_atom_type_list[atom_index_2]

                    if element_1 == element_2 and atom_index_1 > atom_index_2:
                        (atom_index_1, atom_index_2) = (atom_index_2, atom_index_1)
                    elif element_1[0] != 'C' and (element_2[0] == 'C' or mass_dict[element_1[0]] < mass_dict[element_2[0]]):
                        (atom_index_1, atom_index_2) = (atom_index_2, atom_index_1)
                    match = (atom_index_1, atom_index_2, atom_index, subset[2])
                    improper_dihedral_atom_index_list.append(match)

        improper_dihedral_atom_index = torch.tensor(improper_dihedral_atom_index_list, dtype=torch.long)

        return improper_dihedral_atom_index



    def get_dihedral_atom_index(dihedral_pairs, atom_neighbors):

        '''
        Get atom indices of dihedral
        
        Input:
            - dihedral_pairs(tensor): two central atoms on dihedral rotation axis N * 2 
              (eg. (1, 2) in (0, 1, 2, 3))
            - atom_neighbors(tensor): the adjacent node indices of each node
        Output:
            - final_dihedral_atom_index_set(tensor): the position of four atoms building the dihedral M * 4
        '''

        dihedral_pairs_src_node, sort_indices = torch.sort(dihedral_pairs[0])
        dihedral_pairs_dst_node = dihedral_pairs[1][sort_indices]



        dihedral_atom_index_set = []
        for i, src_idx in enumerate(dihedral_pairs_src_node):
            dst_idx = dihedral_pairs_dst_node[i]
            
            src_idx_neighbor = atom_neighbors[src_idx.item()]
            dst_idx_neighbor = atom_neighbors[dst_idx.item()]

            # delete other atom on torsion bond
            
            dst_idx_index = torch.where(src_idx_neighbor == dst_idx)[0].item()
            src_idx_index = torch.where(dst_idx_neighbor == src_idx)[0].item()
            src_idx_neighbor = torch.cat((src_idx_neighbor[:dst_idx_index], 
                                            src_idx_neighbor[dst_idx_index+1:]))
            dst_idx_neighbor = torch.cat((dst_idx_neighbor[:src_idx_index], 
                                            dst_idx_neighbor[src_idx_index+1:]))

            src_neighbor_num = src_idx_neighbor.shape[-1]
            dst_neighbor_num = dst_idx_neighbor.shape[-1]
            if len(src_idx_neighbor.shape) < 2:
                src_idx_neighbor = src_idx_neighbor.unsqueeze(0)
            if len(dst_idx_neighbor.shape) < 2:
                dst_idx_neighbor = dst_idx_neighbor.unsqueeze(0)

            # make dihedral atom tuple like (p0, p1, p2, p3)
            src_idx_set = src_idx.repeat(1, src_neighbor_num)
            src_idx_neighbor = torch.cat((src_idx_neighbor, src_idx_set)).T

            dst_idx_set = dst_idx.repeat(1, dst_neighbor_num)
            dst_idx_neighbor = torch.cat((dst_idx_set, dst_idx_neighbor)).T

            
            src_idx_neighbor = src_idx_neighbor.repeat(dst_neighbor_num, 1)
            dst_idx_neighbor = dst_idx_neighbor.repeat(1, src_neighbor_num)
            dst_idx_neighbor = dst_idx_neighbor.reshape(src_neighbor_num*dst_neighbor_num, 2)

            dihedral_atom_index = torch.cat((src_idx_neighbor, dst_idx_neighbor), dim=1)
            dihedral_atom_index_set.append(dihedral_atom_index)

        # get atom position in dihedral tuple
        dihedral_atom_index_set = torch.cat(dihedral_atom_index_set)
        

        return dihedral_atom_index_set


    # function main part
    dihedral_pairs = get_dihedral_pairs(edge_index, data)
    atom_neighbors = get_neighbors(edge_index)
    dihedral_atom_index_set = get_dihedral_atom_index(dihedral_pairs, atom_neighbors)
    improper_dihedral_atom_index_set = get_improper_dihedral_atom_index(edge_index, data, openmm_atom_type_list)
    final_dihedral_atom_index_set = torch.cat([dihedral_atom_index_set, improper_dihedral_atom_index_set])

    return final_dihedral_atom_index_set


def generate_edge_index(node_index):

    '''
    Generate the edge index according to the node index
    
    Input:
        - node_index(tensor(N,)): the indices of all nodes
    Output:
        - edge_index(tensor(2, M)): the index pairs of source and destination node in edges
    '''
    src_node_index = []
    dst_node_index = []

    num_of_nodes = node_index.shape[0]

    for i in range(num_of_nodes):
        if i == 0:
            src_node_index.append(i)
            dst_node_index.append(i+1)
        elif i == num_of_nodes - 1:
            src_node_index.append(i)
            dst_node_index.append(i-1)
        else:
            src_node_index.append(i)
            src_node_index.append(i)
            dst_node_index.append(i-1)
            dst_node_index.append(i+1)

    edge_index = [src_node_index, dst_node_index]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index




def add_data_force_label(protein_name, residue_name_list, 
                        atom_name_list, pred_coor_list):

    '''
    Generate the edge index according to the node index
    
    Input:
        - protein_name(str): the PDB code or CASP code of protein
        - residue_name_list(list): the name list of residues in protein
        - atom_name_list(list): the name list of atoms in protein
        - pred_coor_list(tensor(N,)): the predicted coordinate list
    Output:
        - data(torch geometric Data object): the graph data with 
        - all_atom_name_list(list): the names of all atoms in the protein
        - all_res_name_list(list): the names of all residues in the protein
        - all_res_index_list(list): the indices of all residues in the protein

    '''
    atom_index_dict = {'C':0,'N':1,'O':2,'S':3}
    atom_type_index_list = []

    bond_length_value_list = []
    angle_value_list = []
    dihedral_value_list = []

    result_bond_index_list = []
    result_angle_index_list = []
    result_dihedral_index_list = []

    protein_atom_coord_dict = add_side_chain_atom(protein_name, residue_name_list, atom_name_list, pred_coor_list)
    all_atom_name_list = protein_atom_coord_dict[protein_name]['atom_name']
    all_res_name_list  = protein_atom_coord_dict[protein_name]['res_name']
    all_res_index_list = protein_atom_coord_dict[protein_name]['res_index']
    all_atom_coord = torch.tensor(protein_atom_coord_dict[protein_name]['coord'], dtype=torch.float)
    CA_atom_index  = torch.tensor(protein_atom_coord_dict[protein_name]['CA_atom_index'], dtype=torch.long)
    protein_edge_index = torch.tensor(protein_atom_coord_dict[protein_name]['edge_index'], dtype=torch.long)

    data = Data(x = all_atom_coord, edge_index = protein_edge_index)
    
    openmm_atom_type_list = []
    for i in range(len(all_atom_name_list)):
        atom_type = all_atom_name_list[i]
        residue_type = all_res_name_list[i]

        openmm_atom_type = atom_type_dict[residue_type][atom_type]
        openmm_atom_type_list.append(openmm_atom_type)

    # record atom type
    for name in all_atom_name_list:
        atom_type_index_list.append(atom_index_dict[name[0]])

    # repair edge index
    start = protein_edge_index[0]
    end = protein_edge_index[1]
    start_idxs, start_vals = torch.unique(start, return_counts=True)
    end_idxs, end_vals = torch.unique(end, return_counts=True)

    if not start_vals.equal(end_vals):
        start_end_diff = start_vals - end_vals
        miss_idx_start = torch.where(start_end_diff < 0)[0]
        end_start_diff = end_vals - start_vals
        miss_idx_end = torch.where(end_start_diff < 0)[0]
        miss_edge_index = torch.cat([miss_idx_start, miss_idx_end]).T.unsqueeze(1)
        protein_edge_index = torch.cat([protein_edge_index, miss_edge_index], dim=1)



    # get bond length info
    src_index = protein_edge_index[0]
    dst_index = protein_edge_index[1]

    for i, index in enumerate(src_index):
        src_atom = openmm_atom_type_list[index]
        dst_atom = openmm_atom_type_list[dst_index[i]]
        try:
            bond_length_value_list.append(bond_force_dict[src_atom][dst_atom])
            result_bond_index_list.append([index, dst_index[i]])
        except:
            continue

    angle_atom_index_list=get_angle_atom_index(protein_edge_index)# get angle value info
    for i, index in enumerate(angle_atom_index_list):
        left_atom = openmm_atom_type_list[index[0]]
        center_atom = openmm_atom_type_list[index[1]]
        right_atom = openmm_atom_type_list[index[2]]
        try:
            angle_value = angle_force_dict[left_atom][center_atom][right_atom]
            angle_value_list.append(angle_value)
            result_angle_index_list.append([index[0], index[1], index[2]])
        except:
            continue

        

    # get dihedral value info
    temp_data = Data(x=all_atom_coord, edge_index=protein_edge_index)
    dihedral_atom_index = get_dihedral_pair_index(temp_data, protein_edge_index, openmm_atom_type_list)
    for i, index in enumerate(dihedral_atom_index):

        atom_1 = openmm_atom_type_list[index[0].item()]
        atom_2 = openmm_atom_type_list[index[1].item()]
        atom_3 = openmm_atom_type_list[index[2].item()]
        atom_4 = openmm_atom_type_list[index[3].item()]
        try:
            dihedral_info_dict = dihedral_force_dict[atom_1][atom_2][atom_3][atom_4]
            result_dihedral_index_list.append([index[0], index[1], index[2], index[3]])
            dihedral_value_list.append([value for value in dihedral_info_dict.values()])
        except Exception as ex:
            continue


    bond_value = torch.tensor(bond_length_value_list, dtype=torch.float)
    angle_value = torch.tensor(angle_value_list, dtype=torch.float)
    dihedral_value = torch.tensor(dihedral_value_list, dtype=torch.float)

    bond_index = torch.tensor(result_bond_index_list, dtype=torch.long)
    angle_index = torch.tensor(result_angle_index_list, dtype=torch.long)
    dihedral_index = torch.tensor(result_dihedral_index_list, dtype=torch.long)

    atom_type_index = torch.tensor(atom_type_index_list, dtype=torch.long)

    data = Data(x=all_atom_coord, CA_atom_index=CA_atom_index, protein_name=protein_name,
                bond_index=bond_index, angle_index=angle_index, dihedral_index=dihedral_index,
                bond_value=bond_value, angle_value=angle_value, dihedral_value=dihedral_value,
                atom_type_index=atom_type_index)

    return data, all_atom_name_list, all_res_name_list, all_res_index_list