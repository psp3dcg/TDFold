'''
Protein Structure Prediction Test File
'''
import os
import sys
import torch
from Bio import SeqIO
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import DataListLoader

import confs.util as util
from data_pipeline import Protein_Dataset
from generate_TDFold_data import preprocess_data
from model.predict_network import Predict_Network
from utils.generate_pdb import generate_pdb, write_pdb
from all_atom_fold.add_force_label import add_data_force_label
from all_atom_fold.energy_minimizer import Energy_Minimizer
def init_name_map():
    #amino acid short name map
    name_map = {"A": "ALA",
                "C": "CYS",
                "D": "ASP",
                "E": "GLU",
                "F": "PHE",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "K": "LYS",
                "L": "LEU",
                "M": "MET",
                "N": "ASN",
                "P": "PRO",
                "Q": "GLN",
                "R": "ARG",
                "S": "SER",
                "T": "THR",
                "V": "VAL",
                "W": "TRP",
                "Y": "TYR"}
    return name_map

def init_atom_num_map():
    #amino acid atom number
    atom_num_map = {"A": 5,
                    "C": 6,
                    "D": 8,
                    "E": 9,
                    "F": 11,
                    "G": 4,
                    "H": 10,
                    "I": 8,
                    "K": 9,
                    "L": 8,
                    "M": 8,
                    "N": 8,
                    "P": 7,
                    "Q": 9,
                    "R": 11,
                    "S": 6,
                    "T": 7,
                    "V": 7,
                    "W": 14,
                    "Y": 12}
    return atom_num_map

def init_atom_name_map():
    #amino acid atom name
    atom_name_map = {"A": ['N','CA','C','O','CB'],
                    "C": ['N','CA','C','O','CB','SG'],
                    "D": ['N','CA','C','O','CB','CG','OD1','OD2'],
                    "E": ['N','CA','C','O','CB','CG','CD','OE1','OE2'],
                    "F": ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
                    "G": ['N','CA','C','O'],
                    "H": ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
                    "I": ['N','CA','C','O','CB','CG1','CG2','CD1'],
                    "K": ['N','CA','C','O','CB','CG','CD','CE','NZ'],
                    "L": ['N','CA','C','O','CB','CG','CD1','CD2'],
                    "M": ['N','CA','C','O','CB','CG','SD','CE'],
                    "N": ['N','CA','C','O','CB','CG','OD1','ND2'],
                    "P": ['N','CA','C','O','CB','CG','CD'],
                    "Q": ['N','CA','C','O','CB','CG','CD','OE1','NE2'],
                    "R": ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
                    "S": ['N','CA','C','O','CB','OG'],
                    "T": ['N','CA','C','O','CB','OG1','CG2'],
                    "V": ['N','CA','C','O','CB','CG1','CG2'],
                    "W": ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
                    "Y": ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH']}
    return atom_name_map

def read_fasta(fasta_file_path):
    '''read fasta

    Input:
        - points_position(tensor):all atom position
    Output:
        - ca_dist(tensor):distance of CA atom pairs
    '''
    result = ""
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        for s in seq_record.seq:
            if s == 'X':
                continue
            result += s
    return result

#dataset split
def data_builder(args, data_path, test_mode, train_gpu_num):

    dataset = Protein_Dataset(data_path, test_mode)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    # import pdb 
    # pdb.set_trace()
    
    if train_gpu_num == 1:
        test_loader = DataLoader(dataset,batch_size=1,shuffle=False)
    elif train_gpu_num > 1:
        test_loader = DataListLoader(dataset,batch_size=1,shuffle=False)

    return test_loader
   

def generate_residue_name(fasta_str, name_map, atom_num_map, atom_mode='bb'):
    '''get amino acid name

    Input:
        - fasta_str(list):fasta string list
        - name_map(dict):amino acid short name dict
        - atom_num_map(dict):amino acid atom number dict
    Output:
        - residue_name_list(list):amino acid name list
        - residue_index_list(list):amino acid index in sequence
    '''
    residue_name_list = []
    residue_index_list = []
    if atom_mode == 'bb':
        res_bb_atom_num = 3
    for i, res in enumerate(list(fasta_str)):
        for j in range(atom_num_map[res]):
            if j >= res_bb_atom_num:
                break
            residue_name_list.append(name_map[res])
            residue_index_list.append(i+1)
    return residue_name_list, residue_index_list

def generate_fasta_and_atom_name(data_path, protein_name, atom_mode='bb'):
    '''get fasta string and atom name

    Input:
        - data_path(str):fasta file path
        - protein_name(str):protein's name
    Output:
        - fasta_str(str):fasta string
        - atom_list(list):all atom in one protein
    '''
    atom_name_map = init_atom_name_map() 

    # fasta string
    fasta_str = read_fasta(os.path.join(data_path, 'raw', protein_name, protein_name+'.fasta'))

    #atom_name_list
    atom_list = []
    if atom_mode == 'bb':
        res_bb_atom_num = 3
    for residue in fasta_str:
        res_atom_list = atom_name_map[residue]
        for i, atom_name in enumerate(res_atom_list):
            if i >= res_bb_atom_num:
                break
            atom_list.append(atom_name)

    return fasta_str, atom_list

def add_side_chain_atom(protein_name, residue_name_list, 
                        atom_name_list, pred_coor_list):
    '''add side chain structure and folding

    Input:
        - protein_name(str): the PDB code of protein
        - residue_name_list(list): the list of residue names
        - atom_name_list(list): the list of atom names
        - pred_coor_list(list): the list of predicted coordinates
    Output:
        - all_atom_coor_list(list): the list of all atom coordinates
        - all_atom_name_list(list): the list of all atom names 
        - all_res_name_list(list): the list of all residue names 
        - all_res_index_list(list): the list of all residue indices
    '''
    all_atom_data = add_data_force_label(protein_name, residue_name_list, 
                        atom_name_list, pred_coor_list)
    data, all_atom_name_list, all_res_name_list, all_res_index_list = all_atom_data
    minimizer = Energy_Minimizer()
    all_atom_coor_list = minimizer.minimize(data)

    return all_atom_coor_list, all_atom_name_list, all_res_name_list, all_res_index_list

def test(args, model, loader, file_path, write_path, lddt_write_path, generate_mode='all atom'):
    '''test function
    '''
    model.eval()
    atom_num_map = init_atom_num_map()
    name_map = init_name_map()
    protein_num_count = 0
    seq_len_list = []
    pLDDT_list = []
    
    for i, data in enumerate(loader):
        print("test %d protein name:%s"%(i, data.protein_name[0]))
        data = data.to(args.device)
        
        
        with torch.no_grad():
            out_coor, pLDDT, _ = model(data, False)
        with open(lddt_write_path, 'a') as f:
            f.write(data.protein_name[0])
            f.write('\n')
            f.write(str(pLDDT.mean().item()*100.)[:4])
            f.write('\n')
        
        
        print('pLDDT value:%.4f'%(pLDDT.mean().item()*100.))
        pLDDT_list.append(pLDDT.mean().item()*100.)

        protein_name = data.protein_name[0]

        fasta_str, atom_name_list = generate_fasta_and_atom_name(file_path, protein_name)

        residue_name_list, residue_index_list = generate_residue_name(fasta_str, name_map, atom_num_map)
        
        seq_len_list.append(len(fasta_str))
        pred_coor_list = out_coor.tolist()
        if generate_mode == 'all atom':
            all_atom_result = add_side_chain_atom(protein_name, residue_name_list, atom_name_list, pred_coor_list)
            all_atom_coor_list, all_atom_name_list, all_res_name_list, all_res_index_list = all_atom_result
            pdb_file_list = generate_pdb(all_atom_name_list, all_atom_coor_list, all_res_name_list, all_res_index_list)
        else:
            pdb_file_list = generate_pdb(atom_name_list, pred_coor_list, residue_name_list, residue_index_list)
        new_write_path = os.path.join(write_path, protein_name+"_pred_result.pdb")
        write_pdb(pdb_file_list, new_write_path)
        
        protein_num_count += 1
        print()

        del data
        torch.cuda.empty_cache()
        with open(lddt_write_path, 'a') as f:
            f.write('\n')
            f.write("The mean value of pLDDT is %.4f"%(sum(pLDDT_list)/len(pLDDT_list)))

def chg_state_dict_name(model, new_state_dict):
    for name, param in model.named_parameters():
        if "attn_L.to_q.bias" in name or "attn_L.to_k.bias" in name or "attn_L.to_v.bias" in name:
            new_state_dict[name] = param * 0.
        elif "attn_N.to_q.bias" in name or "attn_N.to_k.bias" in name or "attn_N.to_v.bias" in name:
            new_state_dict[name] = param * 0.
        elif "attn.to_q.bias" in name or "attn.to_k.bias" in name or "attn.to_v.bias" in name:
            new_state_dict[name] = param * 0.

    new_state_dict["build_graph_model.residue_graph_network.conv1.lin_rel.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv1.lin_l.weight"]
    new_state_dict["build_graph_model.residue_graph_network.conv1.lin_rel.bias"] = new_state_dict["build_graph_model.residue_graph_network.conv1.lin_l.bias"]
    new_state_dict["build_graph_model.residue_graph_network.conv1.lin_root.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv1.lin_r.weight"]
    new_state_dict["build_graph_model.residue_graph_network.conv2.lin_rel.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv2.lin_l.weight"]
    new_state_dict["build_graph_model.residue_graph_network.conv2.lin_rel.bias"] = new_state_dict["build_graph_model.residue_graph_network.conv2.lin_l.bias"]
    new_state_dict["build_graph_model.residue_graph_network.conv2.lin_root.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv2.lin_r.weight"]
    new_state_dict["build_graph_model.residue_graph_network.conv3.lin_rel.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv3.lin_l.weight"]
    new_state_dict["build_graph_model.residue_graph_network.conv3.lin_rel.bias"] = new_state_dict["build_graph_model.residue_graph_network.conv3.lin_l.bias"]
    new_state_dict["build_graph_model.residue_graph_network.conv3.lin_root.weight"] = new_state_dict["build_graph_model.residue_graph_network.conv3.lin_r.weight"]

    return new_state_dict

def chg_env_file():

    env_python_path = sys.executable
    env_path = '/'.join(env_python_path.split('/')[:-2])
    os.system("cp modeling_clip.py {}/lib/python3.10/site-packages/transformers/models/clip/modeling_clip.py".format(env_path))
    os.system("cp safety_checker.py {}/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py".format(env_path))
    
    


def test_func(data_folder_name, result_write_path, SCL_weight_id):
    #parameter initialization
  
    

    parser = util.parser
    args, unknown = parser.parse_known_args()
    torch.manual_seed(args.seed)

    #device selection
    args.device = 'cuda'

    # import pdb
    # pdb.set_trace()

    dataset_name = data_folder_name.split('/')[-1]
    lddt_write_path = os.path.join(result_write_path, "pLDDT_results", dataset_name+'.txt')
    write_path = os.path.join(result_write_path, dataset_name)
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    test_mode = True


    model = Predict_Network(args).to(args.device)

    new_state_dict = torch.load(SCL_weight_id)

    new_state_dict = chg_state_dict_name(model, new_state_dict)

    model.load_state_dict(new_state_dict,strict=False)
    
    test_loader = data_builder(args, data_folder_name, test_mode, 1)
    
    test(args, model, test_loader, data_folder_name, write_path, lddt_write_path)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    fasta_path = sys.argv[1]
    write_base_path = sys.argv[2]
    model_id = sys.argv[3]
    text_lora_id = sys.argv[4] 
    unet_lora_dis_id = sys.argv[5]
    unet_lora_omega_id = sys.argv[6]
    unet_lora_theta_id = sys.argv[7]
    unet_lora_phi_id = sys.argv[8]
    SCL_weight_id = sys.argv[9]
    result_path = sys.argv[10]

    chg_env_file()
    preprocess_data(fasta_path, write_base_path, model_id, text_lora_id, unet_lora_dis_id, unet_lora_omega_id, unet_lora_theta_id, unet_lora_phi_id)
    test_func(write_base_path, result_path, SCL_weight_id)



