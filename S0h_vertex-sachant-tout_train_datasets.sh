#!/bin/bash
#SBATCH --job-name=nested_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_nested_train_datasets_202001016.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_nested_train_datasets_20201016.err
#SBATCH --mem 20000

# vertex sachant tout
python make_classifiers/make_nested_train_dataset.py drugbank_v5.1.5 S0h vertex-sachant-tout --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444 --correct_interactions