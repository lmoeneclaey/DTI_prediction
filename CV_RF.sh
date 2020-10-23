#!/bin/bash
#SBATCH --job-name=CV_rf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/nested_CV_rf_20201015.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/nested_CV_rf_20201015.err
#SBATCH --mem 20000

python cross_validation/RF/cv_RF.py drugbank_v5.1.5 S0h