#!/bin/bash
#SBATCH --job-name=CV_nrlmf_non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_nrlmf_non_balanced_20201023.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_nrlmf_non_balanced_20201023.err
#SBATCH --mem 20000

# python cross_validation/NRLMF/cv_NRLMF.py drugbank_v5.1.5 S0h --center_norm

python cross_validation/NRLMF/nested_cv_NRLMF.py drugbank_v5.1.5 S0h