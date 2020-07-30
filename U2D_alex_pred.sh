#!/bin/bash
#SBATCH --job-name=U2D_Alex_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_Alex_pred_20200728.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_Alex_pred_20200728.err
#SBATCH --mem 20000

# CP7q
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout CP7q --center_norm
# NPPB
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NPPB --center_norm
# NS004
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NS004 --center_norm
# NS1619
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NS1619 --center_norm
# P8
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout P8 --center_norm
# VRT-532
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout VRT-532 --center_norm