#!/bin/bash
#SBATCH --job-name=S0h_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_clf_20200609.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_clf_20200609.err
#SBATCH --mem 20000

python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h VMO --norm --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444