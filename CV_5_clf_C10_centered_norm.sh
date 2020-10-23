#!/bin/bash
#SBATCH --job-name=CV_clf_centered_norm
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_clf_centered_norm_20201015.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_clf_centered_norm_20201015.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_make_kronSVM_clf.py drugbank_v5.1.5 S0h 10 --center_norm