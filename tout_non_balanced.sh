#!/bin/bash
#SBATCH --job-name=tout_non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_non_balanced_20201102.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_non_balanced_20201102.err
#SBATCH --mem 20000

# tout non balanced
python make_classifiers/make_non_balanced_train_dataset.py drugbank_v5.1.5 S0h tout_non_balanced

# tout non balanced
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h tout_non_balanced