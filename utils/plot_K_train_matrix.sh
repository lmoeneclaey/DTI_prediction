#!/bin/bash
#SBATCH --job-name=K_train_matrix
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/K_train_matrix.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/K_train_matrix.err
#SBATCH --mem 80000

python plot_kernels-matrix.py