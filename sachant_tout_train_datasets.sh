#!/bin/bash
#SBATCH --job-name=sachant_tout_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/sachant_tout_train_datasets_20200730.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/sachant_tout_train_datasets_20200730.err
#SBATCH --mem 20000

# ondine
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h sachant-tout