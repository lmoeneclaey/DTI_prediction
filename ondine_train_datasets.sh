#!/bin/bash
#SBATCH --job-name=ondine_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_train_datasets_20200730.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_train_datasets_20200730.err
#SBATCH --mem 20000

# ondine
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h ondine-orphan --orphan DB12442 --orphan DB00304 --orphan DB00294