#!/bin/bash
#SBATCH --job-name=VMO_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_train_datasets_20201019.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_train_datasets_20201019.err
#SBATCH --mem 20000

# vertex modulators orphan
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h vertex-modulators-orphan --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444