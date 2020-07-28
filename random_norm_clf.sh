#!/bin/bash
#SBATCH --job-name=random_center_norm_clf
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/random_center_norm_clf_20200725.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/random_center_norm_clf_20200725.err
#SBATCH --mem 20000

# chloroquine
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h chloroquine-orphan --center_norm

# atovaquone (malarone)
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h atovaquone-orphan --center_norm

# penicillin
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h penicillin-orphan --center_norm

