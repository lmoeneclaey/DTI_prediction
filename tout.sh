#!/bin/bash
#SBATCH --job-name=tout
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_20201029.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/tout_20201029.err
#SBATCH --mem 20000

# tout
# python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h tout

# sachant tout
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h tout