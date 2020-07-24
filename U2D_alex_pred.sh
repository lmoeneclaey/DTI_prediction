#!/bin/bash
#SBATCH --job-name=U2D_Alex_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_Alex_pred_20200708.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/U2D_Alex_pred_20200708.err
#SBATCH --mem 20000

# CP7q
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout CP7q --norm
# NPPB
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NPPB --norm
# NS004
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NS004 --norm
# NS1619
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout NS1619 --norm
# P8
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout P8 --norm
# VRT-532
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout VRT-532 --norm