#!/bin/bash
#SBATCH --job-name=tout_non_balanced_cat3
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/tout/tout_non_balanced_cat3-%A_%a.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/tout/tout_non_balanced_cat3-%A_%a.err
#SBATCH --time=0-100:00
#SBATCH --mem 2000
#SBATCH --ntasks=1
#SBATCH --array=0-233%100
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1

filename="drugs_id_cat3.txt"
drug_nb=$(($SLURM_ARRAY_TASK_ID+1))
drug_id=$(sed "$drug_nb!d" $filename)
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h tout_non_balanced $drug_id