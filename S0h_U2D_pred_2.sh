#!/bin/bash
#SBATCH --job-name=U2D_pred_2
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/S0h_U2D_pred_2_20200623.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/S0h_U2D_pred_2_20200623.err
#SBATCH --mem 20000

## With DrugBank ID

# Bamocaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB15177 --norm
# Alvespimycin
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB12442 --norm
# Digitoxin
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB01396 --norm
# Doxycylin
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB00254 --norm
# Galicaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB14894 --norm
# Posenacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D DB05300 --norm

## Without DrugBank ID

# FDL149-corrector
# python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D FDL149-corrector --norm
# GLPG1837
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D GLPG1837 --norm
# NPPB
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D NPPB --norm
# NS004
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D NS004 --norm
# NS1619
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D NS1619 --norm
# P8
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D P8 --norm
# VRT-532
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h U2D VRT-532 --norm