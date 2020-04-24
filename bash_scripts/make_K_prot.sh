#!/bin/bash
#SBATCH --job-name=Kprot
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/Kprot/Kprot-%A_%a.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/Kprot/Kprot-%A_%a.err
#SBATCH --time=0-100:00
#SBATCH --mem 2000
#SBATCH --ntasks=1
#SBATCH --array=0-2678%100
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1

python process_dataset/make_K_prot.py temp drugbank_v5.1.5 S0h DTI -i $SLURM_ARRAY_TASK_ID