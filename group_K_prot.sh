#!/bin/bash
#SBATCH --job-name=group_Kprot
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/group_K_prot_20201014.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/group_K_prot_20201014.err
#SBATCH --mem 20000

python make_kernels/make_K_prot.py group drugbank_v5.1.5 S0h