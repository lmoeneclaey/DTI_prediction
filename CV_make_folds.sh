#!/bin/bash
#SBATCH --job-name=nested_non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/nested_non_balanced_20201021.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/nested_non_balanced_20201021.err
#SBATCH --mem 20000

python cross_validation/make_folds/nested_cv_make_folds.py drugbank_v5.1.5 S0h 