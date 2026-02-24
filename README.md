## TDFold - Pytorch Implementation

## Folder Organization
├── README.md                 # readme file
├── LICENSE                   # 
├── TDFold_env_requirements.txt # Python environment dependencies of TDFold model
│
├── all_atom_fold/            # the code for refining protein all-atom structure
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后的数据
│   └── examples/             # 示例输入文件
│
├── atom_graph/               # 源代码主目录
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载模块
│   ├── model/                # 模型定义
│   │   ├── __init__.py
│   │   ├── template_diffusion.py  # 模板扩散模型
│   │   └── layers.py         # 网络层定义
│   ├── training/             # 训练相关代码
│   │   ├── train.py          # 主训练脚本
│   │   └── config.py         # 配置文件
│   └── utils/                # 工具函数
│       ├── metrics.py        # 评估指标
│       └── visualization.py  # 可视化工具
│
├── confs/                  # 可执行脚本
│   ├── run_training.sh       # 训练启动脚本
│   ├── run_inference.sh      # 推理启动脚本
│   └── reproduce_results.sh  # 一键复现结果脚本
│
├── examples/                # Jupyter 实验笔记本
│   └── demo.ipynb            # 使用示例
│
├── models/                    # 单元测试
│   ├── test_model.py
│   └── test_data_loader.py
│
├── my_diffusers/                     # 额外文档
│   └── instructions.md       # 详细操作说明
│
├── my_diffusers/                     # 额外文档
│   └── instructions.md       # 详细操作说明
│
├── my_diffusers/                     # 额外文档
│   └── instructions.md       # 详细操作说明
│
├── my_diffusers/                     # 额外文档
│   └── instructions.md       # 详细操作说明
│
└── outputs/                  # 默认输出目录（模型权重、日志等）
    ├── checkpoints/          # 模型检查点
    └── logs/                 # 训练日志

## How to reproduce the experimental results

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
