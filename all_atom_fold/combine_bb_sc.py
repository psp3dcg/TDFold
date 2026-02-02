'''-----combine backbone coord and sidechain coord-----'''
import os
import pdb
import torch
import numpy as np
import numpy.linalg as lin
from Bio.PDB.PDBParser import PDBParser

from utils.aa_info_util import init_reverse_res_name_map
from utils.aa_info_util import init_atom_num_map
from utils.aa_info_util import init_atom_name_map
from utils.aa_info_util import init_aa_edge_index

# the map of amino acid character and short name
res_name_map = init_reverse_res_name_map()

# the number of atoms in each amino acid
atom_num_map = init_atom_num_map()

# amino acid atom name
atom_name_map = init_atom_name_map()

# the indices of source node and destination node of edge (chemical bond) in amino acid 
amino_acid_edge_index = init_aa_edge_index()


def combine_edge_index(residue_list, edge_index_dict):

    '''
    Combine the edge index of each amino acid atom graph
        Input:
            - residue_list(list): the list of residues
            - edge_index_dict(dict): the dict of edge index, 
              the key is the residue name, value is the corresponding edge_index
        Output:
            - combined_edge_index(numpy array): combined edge index of all residues
    '''
    edge_list = []
    total_num_nodes = 0
    last_total_num_nodes = 0

    for i, residue in enumerate(residue_list):
        num_nodes = atom_num_map[residue]
        res_edge_index = np.array(edge_index_dict[residue]).transpose()
        # add the bond connect with the previous residue
        if i > 0:
            link_bond_1 = [total_num_nodes, last_total_num_nodes + 2]
            edge_list.append(link_bond_1)
            last_total_num_nodes = total_num_nodes

        for e_list in res_edge_index:
            # add bond connect the next residue
            if e_list[0] == 3 and e_list[1] == 2 and i != len(residue_list) - 1:
                link_bond_2 = [2 + total_num_nodes, num_nodes + total_num_nodes]
                edge_list.append(link_bond_2)

            e_list[0] = e_list[0] + total_num_nodes
            e_list[1] = e_list[1] + total_num_nodes
            new_e = [e_list[0], e_list[1]]
            edge_list.append(new_e)
        total_num_nodes += num_nodes
    combined_edge_index = np.array(edge_list).transpose()

    return combined_edge_index

def parse_pdb(file_path, seq_id):

    '''
    Parse PDB file
        Input:
            - file_path(str): the list of residues
            - seq_id(str): the sequence (chain) id
        Output:
            - atom_name_list(list): the list of atom names of protein in PDB file
            - atom_coor_list(list): the list of atom coordinates of protein in PDB file
            - res_name_list(list): the list of residue names of protein in PDB file
    '''
    p = PDBParser(PERMISSIVE=1)
    atom_name_list = []
    atom_coor_list = []
    res_name_list = []
    struct_id = "p"

    try:
        s = p.get_structure(struct_id, file_path)
    except:
        print('cannot get struct from pdb')
        return [], []

    model = s[0]
    try:
        chain = model[seq_id]
    except:
        print('cannot get chain {} from pdb'.format(seq_id))
        return [], []

    num_node = 0
    for ch in chain:
        residue = chain[ch.get_id()]
        if ch.get_id()[0] != " ":
            continue
        res_name_list.append(residue.get_resname().upper())
        atom_count = 0
        for res in residue:
            atom = residue[res.get_id()]
            atom_list = str(atom).split(' ')
            new_atom = atom_list[1]
            atom_name = str(new_atom)[0]
            atom_name_with_alpha = str(new_atom)[:-1]
            if atom_name != "H" and atom_name != "G":
                coor_list = list(atom.get_vector())
                atom_name_list.append(atom_name_with_alpha)
                atom_coor_list.append(coor_list)
                atom_count += 1
        
    return atom_name_list, atom_coor_list, res_name_list

def _overlayPoints(points1, points2):

    '''
    Given two sets of points, determine the translation and rotation that matches them as closely as possible.
        Input:
            - points1(numpy array): reference set of coordinates
            - points1(numpy array): set of coordinates to be rotated
        Output:
            - translate2(numpy array): vector to translate points2 by in order to center it
            - rotate(numpy array): rotation matrix to apply to centered points2 to map it on to points1
            - center1(numpy array): center of points1
    '''

    if len(points1) == 0:
        return (np.array(0, 0, 0), np.identity(3), np.array(0, 0, 0))
    if len(points1) == 1:
        return (points1[0], np.identity(3), -1*points2[0])

    # Compute centroids.

    center1 = np.sum(points1, axis=0)/float(len(points1))
    center2 = np.sum(points2, axis=0)/float(len(points2))

    # Compute R matrix.

    R = np.zeros((3, 3))
    for p1, p2 in zip(points1, points2):
        x = p1-center1
        y = p2-center2
        for i in range(3):
            for j in range(3):
                R[i][j] += y[i]*x[j]

    # Use an SVD to compute the rotation matrix.

    (u, s, v) = lin.svd(R)
    return (-1*center2, np.dot(u, v).transpose(), center1)


def read_template_coord(template_path):

    '''
    Read the template amino acid pdb file
        Input:
            - template_path(str): the file path of template amino acid pdb file
        Output:
            - template_res_coord(dict): the dict of template residue (amino acid) coordinates
    '''
    template_res_coord = {}
    for data_name in os.listdir(template_path):
        template_atom_coord = {}
        data_path = os.path.join(template_path, data_name)
        atom_name_list, atom_coord_list, _ = parse_pdb(data_path, 'A')
        for i, name in enumerate(atom_name_list):
            template_atom_coord[name] = atom_coord_list[i]
        template_res_coord[data_name.split('.')[0]] = template_atom_coord
        del template_atom_coord

    return template_res_coord

def get_backbone_coord(protein_name, res_name_list, atom_name_list, atom_coord_list):

    '''
    Select backbone coordinates from all atom coordinates
        Input:
            - protein_name(str): the PDB code of protein
            - res_name_list(list): the list of residue names
            - atom_name_list(list): the list of atom names
            - atom_coord_list(list): the list of atom coordinates
        Output:
            - protein_backbone_info_dict(dict): the residue name and backbone coordinates of protein
    '''
    protein_backbone_info_dict = dict()
    res_coord_list = []
    for i in range(len(res_name_list)//3):
        res_coord_list.append({'res_name':res_name_list[i*3],
                                'bb_atom_coord':{atom_name_list[i*3]:atom_coord_list[i*3],
                                atom_name_list[i*3+1]:atom_coord_list[i*3+1],
                                atom_name_list[i*3+2]:atom_coord_list[i*3+2]}})
    protein_backbone_info_dict[protein_name] = res_coord_list


    return protein_backbone_info_dict



def add_template_coord(protein_backbone_info_dict, template_res_coord_dict):

    '''
    Add template side chain atom coordinates to backbone
        Input:
            - protein_backbone_info_dict(dict): the dict of residue name and backbone coordinates of protein 
            - template_res_coord_dict(dict): the dict of template residue coordinates
        Output:
            - protein_atom_coord_dict(dict): the dict of atom coordinates of protein
    '''
    backbone_name = ('N', 'CA', 'C')
    protein_atom_coord_dict = {}

    for data_key in protein_backbone_info_dict.keys():
        data_info = protein_backbone_info_dict[data_key]
        all_atom_coord_list = []
        all_atom_name_list  = []
        all_res_name_list   = []
        all_res_index_list  = []
        CA_atom_index = []
        res_name_list = []
        res_count = 0
        atom_count = 0
        for res_data in data_info:
            res_count += 1
            templ_backbone_coord_list = []
            backbone_atom_coord_list  = []
            res_atom_coord_list = []
            res_name = res_data['res_name']
            res_name_list.append(res_name_map[res_name])
            templ_res_coord = template_res_coord_dict[res_name]
            backbone_res_coord = res_data['bb_atom_coord']

            for coord in backbone_res_coord.values():
                backbone_atom_coord_list.append(coord)

            for atom_name in atom_name_map[res_name_map[res_name]]:
                if atom_name == 'CA':
                    CA_atom_index.append(atom_count)
                if atom_name in backbone_name:
                    templ_backbone_coord_list.append(templ_res_coord[atom_name])
                all_atom_name_list.append(atom_name)
                all_res_name_list.append(res_name)
                all_res_index_list.append(res_count)
                # import pdb 
                # pdb.set_trace()
                try:
                    res_atom_coord_list.append(templ_res_coord[atom_name])
                except:
                    print(res_name)
                    print(res_name_map[res_name])
                    print(atom_name_map[res_name_map[res_name]])
                    print(atom_name)


                atom_count += 1
            # compute rotate and trans matrix per residue
            (translate2, rotate, translate1) = _overlayPoints(backbone_atom_coord_list, templ_backbone_coord_list)

            for idx, atom_coord in enumerate(res_atom_coord_list):
                if idx < 3:
                    all_atom_coord_list.append(backbone_atom_coord_list[idx])
                else:
                    all_atom_coord_list.append(np.array(np.dot(rotate,atom_coord+translate2))+translate1)


        protein_edge_index = combine_edge_index(res_name_list, amino_acid_edge_index)
        protein_atom_coord_dict[data_key] = {'coord':all_atom_coord_list,
                                            'atom_name':all_atom_name_list,
                                            'res_name':all_res_name_list,
                                            'res_index':all_res_index_list,
                                            'CA_atom_index':CA_atom_index,
                                            'edge_index':protein_edge_index}


    return protein_atom_coord_dict



def add_side_chain_atom(protein_name, res_name_list, 
                        atom_name_list, atom_coord_list):
    
    '''
    Given two sets of points, determine the translation and rotation that matches them as closely as possible.
        Input:
            - protein_name(str): the PDB code of protein
            - res_name_list: the list of residue names
            - atom_name_list: the list of atom names
            - atom_coord_list: the list of atom coordinates
        Output:
            - protein_atom_coord_dict: the residue name and backbone coordinates of protein
    '''
    template_path = "all_atom_fold/unfold_aa/"
    template_res_coord_dict = read_template_coord(template_path)
    protein_backbone_info_dict = get_backbone_coord(protein_name, res_name_list, atom_name_list, atom_coord_list)
    protein_atom_coord_dict = add_template_coord(protein_backbone_info_dict, template_res_coord_dict)

    return protein_atom_coord_dict

