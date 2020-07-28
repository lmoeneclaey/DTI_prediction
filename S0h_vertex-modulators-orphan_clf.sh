#!/bin/bash
#SBATCH --job-name=VMO_center_norm_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-modulators-orphan_center_norm_clf_20200726.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-modulators-orphan_center_norm_clf_20200726.err
#SBATCH --mem 20000

python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h vertex-modulators-orphan --center_norm