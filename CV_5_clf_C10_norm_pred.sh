#!/bin/bash
#SBATCH --job-name=CV_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_norm_pred_20200724.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_norm_pred_20200724.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_kronSVM_pred.py drugbank_v5.1.5 S0h 5 10 --norm