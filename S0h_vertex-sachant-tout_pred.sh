#!/bin/bash
#SBATCH --job-name=U2D_center_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-sachant-tout_center_norm_pred_20200726.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-sachant-tout_center_norm_pred_20200726.err
#SBATCH --mem 20000

# Vertex
## Ivacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB08820 --center_norm
## Tezacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB11712 --center_norm
## Lumacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB09280 --center_norm
## Elexacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB15444 --center_norm
## Bamocaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout VX-659 --center_norm

# Galapgaos
## GLPG1837
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG1837 --center_norm
## GLPG2222
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG2222 --center_norm

# Proteostatis
## PTI-428
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-428 --center_norm
## PTI-801
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-801 --center_norm
## PTI-808
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-808 --center_norm

## FDL169
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout FDL169 --center_norm