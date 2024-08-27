#!/bin/bash

sbatch -N 1 --gres=gpu:1 -p vip_gpu_ailab -A ai4phys train.sh
