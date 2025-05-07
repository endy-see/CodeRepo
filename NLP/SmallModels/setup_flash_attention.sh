#!/bin/bash

# 确保CUDA环境变量设置正确
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 创建新的conda环境
conda create -n flash_attn python=3.10 -y
conda activate flash_attn

# 安装PyTorch (使用conda确保CUDA兼容性)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装必要的构建工具
conda install -y ninja

# 安装flash-attention的依赖
pip install packaging

# 克隆并安装flash-attention
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention

# 安装flash-attention (会自动处理CUDA兼容性)
pip install . --no-build-isolation

# 安装transformers和其他必要的包
pip install transformers accelerate tensorboard scikit-learn pandas numpy tqdm

# 验证安装
python -c "import flash_attn; print('Flash attention installed successfully!')"

echo "Setup complete! Please make sure to activate the environment with: conda activate flash_attn"