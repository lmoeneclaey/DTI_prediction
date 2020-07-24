#!/bin/bash
#SBATCH --job-name=U2D_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_train_datasets_20200723.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_train_datasets_20200723.err
#SBATCH --mem 20000

# vertex sachant tout
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h vertex-sachant-tout --norm --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444 --correct_interactions