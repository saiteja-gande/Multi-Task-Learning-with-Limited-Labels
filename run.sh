#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output trainmodel.out

python main.py --weighting EW --arch TIT --dataset_path /dataset --gpu_id 0 --scheduler step --aug --epochs 150 --save_path /save --ulw 7 --pld 10 --task Seg