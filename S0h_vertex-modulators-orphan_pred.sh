#!/bin/bash
#SBATCH --job-name=VMO_norm_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-modulators-orphan_norm_pred_20200709.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV/vertex-modulators-orphan_norm_pred_20200709.err
#SBATCH --mem 20000

# Vertex
## Ivacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB08820 --norm
## Tezacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB11712 --norm
## Lumacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB09280 --norm
## Elexacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB15444 --norm
## Bamocaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan VX-659 --norm

# Galapgaos
## GLPG1837
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan GLPG1837 --norm
## GLPG2222
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan GLPG2222 --norm

# Proteostatis
## PTI-428
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-428 --norm
## PTI-801
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-801 --norm
## PTI-808
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-808 --norm

# FDL169
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan FDL169 --norm
