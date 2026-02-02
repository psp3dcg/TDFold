import os
import sys
from generate_pt_file import data_builder
from atom_graph.generate_atom_data import generate_atom_graph_data
from generate_inter_residue_geometry import InterResidueGeometryGenerator


def preprocess_data(fasta_path, write_base_path, model_id, text_lora_id, unet_lora_dis_id, unet_lora_omega_id, unet_lora_theta_id, unet_lora_phi_id):

    data_name = write_base_path
    write_path = os.path.join(write_base_path, 'raw')
    if not os.path.exists(write_base_path):
        os.mkdir(write_base_path)
    
    if not os.path.exists(write_path):
        os.mkdir(write_path)
	
	    
	
    print("--------1.generating atom graph files...")
    generate_atom_graph_data(fasta_path, write_path)
    print()


    print("--------2.generating pt files...")
    data_builder(data_name)
    print()

    print("--------3.generating inter-residue geometries...")
    generator = InterResidueGeometryGenerator(model_id, text_lora_id, unet_lora_dis_id, unet_lora_omega_id, unet_lora_theta_id, unet_lora_phi_id)
    generator.generate_inter_residue_geometry(fasta_path, os.path.join(data_name, 'processed'))
    print()

    print("--------generating process is finished...")

if __name__ == '__main__':

    fasta_path = sys.argv[1]
    write_base_path = sys.argv[2]
    model_id = sys.argv[3]
    text_lora_id = sys.argv[4] 
    unet_lora_dis_id = sys.argv[5]
    unet_lora_omega_id = sys.argv[6]
    unet_lora_theta_id = sys.argv[7]
    unet_lora_phi_id = sys.argv[8]
    preprocess_data(fasta_path, write_base_path, model_id, text_lora_id, unet_lora_dis_id, unet_lora_omega_id, unet_lora_theta_id, unet_lora_phi_id)
