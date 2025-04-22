#!/bin/bash
#SBATCH -N 1
# SBATCH -t 00:02:00
#SBATCH -t 00:10:00
#SBATCH -c 1
#SBATCH --partition=lake-gpu
# SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lkm322@lehigh.edu

#conda activate cenv_pytorch1

# for i in $(seq 10 5 55); do
# for i in $(seq 1000 125 2000); do
# for i in $(seq 1000 50 1250); do
# for i in $(seq 1625 125 2000); do
# for i in $(seq 1800 1800); do
# for i in $(seq 1000 1000); do
# for i in $(seq 18 18); do
python3 FV_RP_explicit_unstructured_RHLL_supersonicCylinders_Param.py 1125;
# python3 FV_RP_explicit_unstructured_RHLL_supersonicCylinders_Param.py $i;
# done;
