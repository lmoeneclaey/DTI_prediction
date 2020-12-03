#!/bin/bash
#SBATCH --job-name=rosco-orphan
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/rosco-orphan_20201111.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/rosco-orphan_20201111.err
#SBATCH --mem 20000

# Train datasets
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h roscovitine-orphan --orphan DB06195

# Classifier
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h roscovitine-orphan

# Prediction
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h roscovitine-orphan DB06195
