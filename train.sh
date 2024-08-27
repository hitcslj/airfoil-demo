#!/bin/bash
#SBATCH --partition=vip_gpu_ailab     # 指定分区（可选）
#SBATCH --account=ai4phys     # 指定账户名称
#SBATCH --nodes=1               # 使用的节点数
#SBATCH --ntasks=1              # 任务数
#SBATCH --gres=gpu:1              # GPU 数量

#SBATCH --output=./logs/sbatch_airfoil_demo.out     # 标准输出日志文件
#SBATCH --error=./logs/sbatch_airfoil_demo.err      # 标准错误日志文件

module load anaconda/2021.11 
module unload cudnn
#module load compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
module load  compilers/cuda/11.8    cudnn/8.6.0.163_cuda11.x
module load compilers/gcc/11.3.0

#conda init bash
source /home/bingxing2/apps/anaconda/2021.11/etc/profile.d/conda.sh
conda activate airfoil

python -u demo_diffusion.py
