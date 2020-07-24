#!/bin/bash
#SBATCH --job-name=random_train_datasets
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/random_train_datasets_20200723.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/random_train_datasets_20200723.err
#SBATCH --mem 20000

# chloroquine
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h chloroquine-orphan --orphan DB00608

# atovaquone (malarone)
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h atovaquone-orphan --orphan DB01117

# penicillin
python make_classifiers/make_train_dataset.py drugbank_v5.1.5 S0h penicillin-orphan --orphan DB01053