#!/bin/bash
#SBATCH --job-name=U2D_norm_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_norm_clf_20200707.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vertex-sachant-tout_norm_clf_20200707.err
#SBATCH --mem 20000

python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h vertex-sachant-tout --norm --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444 --correct_interactions