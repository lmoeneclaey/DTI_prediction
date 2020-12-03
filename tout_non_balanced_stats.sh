#!/bin/bash
#SBATCH --job-name=tout_non_balanced_stats
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_20201113/tout_non_balanced_stats-%A_%a.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_20201113/tout_non_balanced_stats-%A_%a.err
#SBATCH --time=0-100:00
#SBATCH --mem 2000
#SBATCH --ntasks=1
#SBATCH --array=0-5070%100
#SBATCH --nodes=5
##SBATCH --cpus-per-task=1

filename="drugs_id.txt"
drug_nb=$(($SLURM_ARRAY_TASK_ID+1))
drug_id=$(sed "$drug_nb!d" $filename)
python get_all_predictions_stats.py drugbank_v5.1.5 S0h tout_non_balanced $drug_id