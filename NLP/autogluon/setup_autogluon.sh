#!/bin/bash

# Step 1: Create a new conda environment with Python 3.10
conda create -n autogluon_env python=3.10 -y

# Step 2: Activate the environment
source activate autogluon_env

# Step 3: Install PyTorch dependencies first
conda install -y sympy=1.13.1
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 4: Install CUDA toolkit compatibility packages
conda install -y cudatoolkit-dev=12.1 -c conda-forge

# Step 5: Install key dependencies using conda to ensure binary compatibility
conda install -y \
    pandas=2.0.0 \
    numpy=1.25.0 \
    scipy \
    scikit-learn=1.4.0 \
    pillow=10.0.1 \
    boto3 \
    requests \
    matplotlib \
    -c conda-forge

# Step 6: Install HuggingFace packages first
pip install --no-deps tokenizers
pip install \
    'transformers' \
    'datasets' \
    'huggingface-hub>=0.19.0'

# Step 7: Install base dependencies with pip
pip install \
    'threadpoolctl>=3.1.0' \
    'psutil>=5.7.3' \
    'joblib>=1.0.1' \
    'nltk>=3.4.5,<3.9'

# Step 8: Install AutoGluon dependencies
pip install \
    'accelerate>=0.34.0,<1.0' \
    'defusedxml>=0.7.1,<0.7.2' \
    'evaluate>=0.4.0,<0.5.0' \
    'lightning>=2.2,<2.6' \
    'nlpaug>=1.1.10,<1.2.0' \
    'nvidia-ml-py3==7.352.0' \
    'omegaconf>=2.1.1,<2.3.0' \
    'openmim>=0.3.7,<0.4.0' \
    'pdf2image>=1.17.0,<1.19' \
    'pytesseract>=0.3.9,<0.3.11' \
    'pytorch-metric-learning>=1.3.0,<2.4' \
    'scikit-image>=0.19.1,<0.25.0' \
    'seqeval>=1.2.2,<1.3.0' \
    'text-unidecode>=1.3,<1.4' \
    'timm>=0.9.5,<1.0.7' \
    'torchmetrics>=1.2.0,<1.3.0'

# Step 9: Install AutoGluon
pip install autogluon

# Step 10: Install additional ML packages
conda install -y \
    lightgbm \
    catboost \
    xgboost \
    -c conda-forge

pip install \
    'optuna>=3.0.0' \
    'ray>=2.0.0' \
    'tensorboard>=2.11.0'

# Step 11: Verify installation and show CUDA information
echo "Verifying installation..."
python -c "
import torch
import autogluon.multimodal
import transformers
import pandas as pd
import numpy as np
import sympy

print('System CUDA information:')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Current CUDA device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))

print('\nPackage versions:')
print('AutoGluon version:', autogluon.__version__)
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
print('Pandas version:', pd.__version__)
print('Numpy version:', np.__version__)
print('Sympy version:', sympy.__version__)
"

echo "Installation completed. Please run 'conda activate autogluon_env' to use the environment."

# Note: If you encounter CUDA compatibility issues, you may need to set:
echo "If needed, set these environment variables:"
echo "export CUDA_HOME=/usr/local/cuda-12.2"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64"
