#!/bin/bash
#SBATCH --job-name=VMO_alex_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_alex_pred_20200709.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_alex_pred_20200709.err
#SBATCH --mem 20000

# CP7q
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan CP7q --norm
# NPPB
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan NPPB --norm
# NS004
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan NS004 --norm
# NS1619
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan NS1619 --norm
# P8
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan P8 --norm
# VRT-532
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan VRT-532 --norm
