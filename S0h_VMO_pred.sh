#!/bin/bash
#SBATCH --job-name=VMO_non_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_VMO_non_norm_pred_20200619.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_VMO_non_norm_pred_20200619.err
#SBATCH --mem 20000

# Ivacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB08820
# Tezacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB11712
# Lumacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB09280
# Elexacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB15444