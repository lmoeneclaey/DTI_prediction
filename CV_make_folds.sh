#!/bin/bash
#SBATCH --job-name=CV_make_folds
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_make_folds_20200723.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_make_folds_20200723.err
#SBATCH --mem 20000

python cross_validation/make_folds/cv_make_folds.py drugbank_v5.1.5 S0h 5