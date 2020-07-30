#!/bin/bash
#SBATCH --job-name=CV_nrlmf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_nrlmf_20200730.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/CV_nrlmf_20200730.err
#SBATCH --mem 20000

python cross_validation/NRLMF/cv_NRLMF.py drugbank_v5.1.5 S0h --center_norm