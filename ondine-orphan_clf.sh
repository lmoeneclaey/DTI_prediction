#!/bin/bash
#SBATCH --job-name=ondine_center_norm_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_center_norm_clf_20200730.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_center_norm_clf_20200730.err
#SBATCH --mem 20000

# orphan
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h ondine-orphan --center_norm