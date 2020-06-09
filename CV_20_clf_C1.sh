#!/bin/bash
#SBATCH --job-name=CV_C1
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C1_20200608.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_C1_20200608.err
#SBATCH --mem 20000

python cross_validation/kronSVM/cv_make_kronSVM_clf.py drugbank_v5.1.5 S0h 20 1 --norm