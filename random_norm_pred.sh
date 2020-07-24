#!/bin/bash
#SBATCH --job-name=random_center_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/random_center_norm_pred_20200717.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/random_center_norm_pred_20200717.err
#SBATCH --mem 20000

# chloroquine
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h chloroquine-orphan DB00608 --center_norm 

# atovaquone (malarone)
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h atovaquone-orphan DB01117 --center_norm

# penicillin
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h penicillin-orphan DB01053 --center_norm

