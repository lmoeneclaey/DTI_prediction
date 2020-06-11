#!/bin/bash
#SBATCH --job-name=S0h_CMO_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_VMO_pred_20200609.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/S0h_VMO_pred_20200609.err
#SBATCH --mem 20000

# Ivacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB08820 --norm
# Tezacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB11712 --norm
# Lumacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB09280 --norm
# Elexacaftor
python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB15444 --norm