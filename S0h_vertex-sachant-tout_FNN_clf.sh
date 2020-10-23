#!/bin/bash
#SBATCH --job-name=U2D_FNN_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_FNN_clf_20201019.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_FNN_clf_20201019.err
#SBATCH --mem 20000

python make_classifiers/FNN_clf/make_FNN_clf.py drugbank_v5.1.5 S0h vertex-sachant-tout