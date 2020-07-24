#!/bin/bash
#SBATCH --job-name=U2D_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-sachant-tout_norm_pred_20200709.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-sachant-tout_norm_pred_20200709.err
#SBATCH --mem 20000

# Vertex
## Ivacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB08820 --norm
## Tezacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB11712 --norm
## Lumacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB09280 --norm
## Elexacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB15444 --norm
## Bamocaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout VX-659 --norm

# Galapgaos
## GLPG1837
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG1837 --norm
## GLPG2222
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG2222 --norm

# Proteostatis
## PTI-428
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-428 --norm
## PTI-801
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-801 --norm
## PTI-808
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-808 --norm

## FDL169
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout FDL169 --norm