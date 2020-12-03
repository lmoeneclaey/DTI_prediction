#!/bin/bash
#SBATCH --job-name=group_tout
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/group_tout_20201113.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/group_tout_20201113.err
#SBATCH --mem 20000

# tout non balanced
python group_all_predictions_stats.py drugbank_v5.1.5 S0h tout