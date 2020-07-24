#!/bin/bash
#SBATCH --job-name=CV_clf_norm
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_clf_norm_20200724.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_clf_norm_20200724.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_make_kronSVM_clf.py drugbank_v5.1.5 S0h 5 10 --norm