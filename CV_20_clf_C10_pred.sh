#!/bin/bash
#SBATCH --job-name=CV_C10_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C10_pred_20200609.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C10_pred_20200609.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_kronSVM_pred.py drugbank_v5.1.5 S0h 20 10 --norm