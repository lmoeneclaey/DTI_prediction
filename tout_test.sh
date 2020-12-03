#!/bin/bash
#SBATCH --job-name=tout_test
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_test-%A_%a.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_test-%A_%a.err
#SBATCH --time=0-100:00
#SBATCH --mem 2000
#SBATCH --ntasks=1
#SBATCH --array=0-3%2
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1

filename="4_drugs.txt"
drug_nb=$(($SLURM_ARRAY_TASK_ID+1))
drug_id=$(sed "$drug_nb!d" $filename)
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h tout $drug_id