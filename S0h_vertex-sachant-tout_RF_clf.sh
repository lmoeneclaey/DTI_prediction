#!/bin/bash
#SBATCH --job-name=U2D_RF_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_RF_clf_20201015.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_RF_clf_20201015.err
#SBATCH --mem 20000

python make_classifiers/RF_clf/make_RF_clf.py drugbank_v5.1.5 S0h vertex-sachant-tout