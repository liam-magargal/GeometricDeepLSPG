#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --partition=lake-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lkm322@lehigh.edu

# activate conda environment
# conda activate cenv_pytorch1

# run
python Cyl_Hierarchical_Mesh_Generator.py
#python -u FV_Burgers_CNN_train_1.py > CNN_train_1.out
#python -u test1.py > CNN_train_1.out
