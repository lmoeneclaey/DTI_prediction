#!/bin/bash
#SBATCH --job-name=VMO_SVM_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-modulators-orphan_SVM_clf_20201020.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-modulators-orphan_SVM_clf_20201020.err
#SBATCH --mem 20000

python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h vertex-modulators-orphan --center_norm