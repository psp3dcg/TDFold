## TDFold - Pytorch Implementation

## Folder Organization
```
├── LICENSE                            # License file
├── README.md                          # Project documentation
├── TDFold_env_requirements.txt        # Python dependencies of TDFold model
├── data_pipeline.py                   # Datatype defination of torch geometric
├── generate_TDFold_data.py            # Main generating process of protein data
├── generate_inter_residue_geometry.py # Generating process of 2D geometric template
├── generate_TDFold_data.py            # Main generating process of protein data
├── generate_pt_file.py                # Generating process of protein graph data
├── modeling_clip.py                   # New modeling_clip to make text encoder support longer text prompt (>77 tokens)
├── safety_checker.py                  # New safety_checker to avoid stable diffusion model'check for prompt
├── generate_TDFold_data.py            # Main generating process of protein data
│
├── all_atom_fold/                     # Refining protein all-atom structure
│   ├── __init__.py/                   # 
│   ├── add_force_label.py/            # Adding the force information to protein graph data
│   ├── combine_bb_sc.py/              # Combining the backbone and sidechain
│   ├── energy_minimizer.py/           # Energy minimizer
│   ├── loss_func.py/                  # Harmonic energy loss function
│   ├── force_file/                    # Force information
│   └── unfold_aa                      # Unfolded structures of 20 amino acids
│
├── atom_graph/                        # Building the torch geometric atom-level graph data for protein
│   ├── __init__.py/                   # 
│   ├── add_bond_type.py/              # Adding the type of atom bond
│   ├── generate_atom_data.py/         # Building process of atom-level graph data
│   ├── generate_protein_graph.py/     # Building-relative functions of grpah data
│   └── amino acid                     # The information (e.g., amino acid type) of 20 amino acids
│
├── confs/                             # Training configuration
│   ├── __init__.py/                   # 
│   └── util.py                        # Training configuration parameters
│
├── examples/                          # The generation example python files of stable diffusion models
│
├── model/                             # The python files of sequence-geometry collaborative learning (SCL) model
│   ├── __init__.py/                   #
│   ├── Attention_module_w_str.py/     # Attention computation used in transformer of TDFold
│   ├── DistancePredictor.py/          # Predicting the inter-residue distance and orientations of the input protein
│   ├── Embeddings.py/                 # Making embeddings of the protein sequence
│   ├── InitStrGenerator.py/           # Predicting the initial backbone structure of protein
│   ├── LDDT_torch.py/                 # Pytorch implementation of local distance difference test (LDDT)
│   ├── SE3_network.py/                # SE(3)-EGNN for refining protein 3D structure
│   ├── Transformer.py/                # Transformer model used in TDFold
│   ├── aa_info_util.py/               # 20 amino acids information
│   ├── build_graph.py/                # Building the residue-level graph data and making residue and atom feature fusion
│   ├── kinematics.py/                 # Transforming the coordinates to distances and orientations
│   ├── parse_protein_info.py/         # Parsing the protein sequence and 2D geometric template information
│   ├── parsers.py/                    # homology parser used in the version of TDFold with MSA
│   ├── predict_network.py/            # Main process of protein structure prediction 
│   ├── refine_net.py/                 # Refining the backbone structure with SE3 network
│   ├── resnet.py/                     # Unsymmetric CNN used in hybrid CNN
│   ├── rigid_transform.py/            # Rigid alignment of predicted and true protein structure before computing RMSD loss
│   └── sym_cnn.py                     # Symmetric CNN in hybrid CNN
│
├── my_diffusers/                      # The pytorch implementation of stable diffusion model
│
├── scripts/                           # The script files of stable diffusion model
│
├── src/                               # The python file of diffusers
│
├── tests/                             # The test python files of stable diffusion model
│
└── utils/                             # 
    ├── aa_info_util.py/               # amino acid information
    ├── equivariant_attention/         # Equivariant attentions of SE3-EGNN
    ├── generate_pdb.py/               # Generating pdb file based on predicted coordinates
    └── loss_func.py/                  # Training loss functions of TDFold
```  

## Instructions on code implementation for reproducing results

## 1. Installing the conda environment
---------------
1) Create conda environment using `TDFold_env_requirements.txt` file
```
conda env create -f TDFold_env python=3.10
pip install -r TDFold_env_requirements.txt
```

## 2. Downloading the model parameter file and replacing model library file of conda environment
---------------
1) Download the stable diffusion (SD) model parameters from [stable-diffusion](https://github.com/CompVis/stable-diffusion) and the LoRA parameters from [Zenodo](https://zenodo.org/records/18530072).

2) Replace the library file of conda environment for clip model
```
cp your_env_path/transformers/models/clip/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip_bp.py
cp TDFold_code/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip.py
```

3) Replace the library file of conda environment for safety checker
```
cp your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker_bp.py
cp TDFold_code/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py
```



## 3. Prediction usage example
---------------
1) The python command of running the model to predict the protein structures:
```
python -u test_TDFold.py [FASTA_folder_path] [data_write_path] [SD_model_path] [text_encoder_LoRA_model_path] [UNet_dis_LoRA_model_path] [UNet_omega_LoRA_model_path] [UNet_theta_LoRA_model_path] [UNet_phi_LoRA_model_path] [SCL_model_path] [result_path]

-FASTA_folder_path: the path of folder containing FASTA files
-data_write_path: the path of folder to write generated data
-SD_model_path: the path of Stable Diffusion pretrained model
-text_encoder_LoRA_model_path: the path of text LoRA model
-UNet_dis_LoRA_model_path: the path of UNet LoRA model for pairwise CB distance matrix
-UNet_omega_LoRA_model_path: the path of UNet LoRA model for pairwise omega dihedral matrix
-UNet_theta_LoRA_model_path: the path of UNet LoRA model for pairwise theta dihedral matrix
-UNet_phi_LoRA_model_path: the path of UNet LoRA model for pairwise phi angle matrix
-SCL_model_path: the path of sequence-geometry collaborative learning (SCL) model parameter file
-result_path: the path of output structure results
```
