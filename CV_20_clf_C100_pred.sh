#!/bin/bash
#SBATCH --job-name=CV_C100_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C100_pred_20200608.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C100_pred_20200608.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_kronSVM_pred.py drugbank_v5.1.5 S0h 20 100 --norm