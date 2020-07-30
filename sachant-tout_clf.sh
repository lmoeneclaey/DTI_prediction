#!/bin/bash
#SBATCH --job-name=sachant-tout_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/sachant-tout_clf_20200730.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/sachant-tout_clf_20200730.err
#SBATCH --mem 20000

# sachant tout
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h sachant-tout --center_norm