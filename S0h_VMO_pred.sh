#!/bin/bash
#SBATCH --job-name=S0h_VMO_pred
#SBATCH --nodes=1
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/S0h_VMO/S0h_VMO.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/S0h_VMO/S0h_VMO.err
#SBATCH --mem 10000

# Tezacaftor
PYTHONPATH=/mnt/data4/mnajm/ python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB11712 --norm
# Lumacaftor
PYTHONPATH=/mnt/data4/mnajm/ python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB09280 --norm
# Elexacaftor
PYTHONPATH=/mnt/data4/mnajm/ python predict/kronSVM/kronSVM_pred.py drugbank_v5.1.5 S0h VMO DB15444 --norm