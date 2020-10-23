#!/bin/bash
#SBATCH --job-name=U2D_SVM_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_SVM_clf_20201015.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_SVM_clf_20201015.err
#SBATCH --mem 20000

python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h vertex-sachant-tout --center_norm