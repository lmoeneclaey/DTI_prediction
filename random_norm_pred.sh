#!/bin/bash
#SBATCH --job-name=random_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/random_pred_20201019.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/random_pred_20201019.err
#SBATCH --mem 20000

# chloroquine
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB00608 --center_norm 
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB00608

# atovaquone (malarone)
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB01117 --center_norm
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB01117

# penicillin
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB01053 --center_norm
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB01053
