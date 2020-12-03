#!/bin/bash
#SBATCH --job-name=concat_tout_non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/concat_tout_non_balanced_20201113.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/concat_tout_non_balanced_20201113.err
#SBATCH --mem 20000

# tout non balanced
python concat_all_predictions.py drugbank_v5.1.5 S0h tout_non_balanced