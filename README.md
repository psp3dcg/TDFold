## TDFold - Pytorch Implementation

## Folder Organization
```
├── README.md                # Project documentation
├── LICENSE                  # License file
├── TDFold_env_requirements.txt         # Python dependencies
│
├── data/                    # Dataset files
│   ├── examples/           # Example input files
│   └── README.md           # Data description
│
├── src/                     # Source code
│   ├── models/             # Model architectures
│   ├── utils/              # Helper functions
│   └── config.py           # Configuration settings
│
├── scripts/                 # Run scripts
│   ├── train.sh            # Training script
│   ├── predict.sh          # Inference script
│   └── reproduce.sh        # Reproduction script
│
├── notebooks/               # Jupyter notebooks
│   └── demo.ipynb          # Usage demonstration
│
└── outputs/                 # Generated results
    ├── checkpoints/        # Model weights
    └── logs/               # Training logs
```  

## Instructions on code implementation for reproducing results

## 1. Installing the conda environment
---------------
1) Create conda environment using `TDFold_env_requirements.txt` file
```
conda env create -f TDFold_env python=3.10
pip install -r TDFold_env_requirements.txt
```

## 2. Replacing the conda environment library file
---------------
1) Replace the clip model library file
```
cp your_env_path/transformers/models/clip/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip_bp.py
cp TDFold_code/modeling_clip.py your_env_path/transformers/models/clip/modeling_clip.py
```

2) Replace the stable diffusion model library file
```
cp your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker_bp.py
cp TDFold_code/safety_checker.py your_env_path/diffusers/pipelines/stable_diffusion/safety_checker.py
```

3) Download the stable diffusion (SD) model parameters from [stable-diffusion](https://github.com/CompVis/stable-diffusion) and the LoRA parameters from [Zenodo](https://zenodo.org/records/18530072).

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
