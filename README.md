## TDFold - Pytorch

## Installation
---------------
1. Create conda environment using `TDFold_env_requirements.txt` file
```
conda env create -f TDFold_env python=3.10
pip install -r TDFold_env_requirements.txt
```

## Library file replacement
---------------
1. replace the clip model library file
```
cp your_env_path/transformers/models/clip/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip_bp.py
cp TDFold_code/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip.py
```

2. replace the stable diffusion model library file
```
cp your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker_bp.py
cp TDFold_code/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py
```

## Usage example
---------------
1. Predict the protein structures:
```
python -u test_TDFold.py [FASTA_folder_path] [data_write_path] [SD_model_path] [text_encoder_LoRA_model_path] [UNet_dis_LoRA_model_path] [UNet_omega_LoRA_model_path] [UNet_theta_LoRA_model_path] [UNet_phi_LoRA_model_path] [SCL_model_path] [result_path]

-FASTA_folder_path: the path of folder containing FASTA files
-data_write_path: the path of folder to write generated data
-SD_model_path: the path of Stable Diffusion pretrained model
-text_encoder_LoRA_model_path: the path of pdb100 dataset
-UNet_dis_LoRA_model_path: the path of UNet LoRA model for pairwise CB distance matrix
-UNet_omega_LoRA_model_path: the path of UNet LoRA model for pairwise omega dihedral matrix
-UNet_theta_LoRA_model_path: the path of UNet LoRA model for pairwise theta dihedral matrix
-UNet_phi_LoRA_model_path: the path of UNet LoRA model for pairwise phi angle matrix
-SCL_model_path: the path of SCL model file
-result_path: the path of output results
```
