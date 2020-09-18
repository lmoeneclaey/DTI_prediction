#!/bin/bash
#SBATCH --job-name=VMO_NRLMF
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_NRLMF_20200901.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/VMO_NRLMF_20200901.err
#SBATCH --mem 20000

# Vertex
## Ivacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB08820 --center_norm
## Tezacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB11712 --center_norm
## Lumacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB09280 --center_norm
## Elexacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan DB15444 --center_norm
## Bamocaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan VX-659 --center_norm

# Galapgaos
## GLPG1837
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan GLPG1837 --center_norm
## GLPG2222
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan GLPG2222 --center_norm

# Proteostatis
## PTI-428
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-428 --center_norm
## PTI-801
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-801 --center_norm
## PTI-808
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan PTI-808 --center_norm

# FDL169
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-modulators-orphan FDL169 --center_norm