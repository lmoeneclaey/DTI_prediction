#!/bin/bash
#SBATCH --job-name=VST_RF_pred
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/VST_RF_pred_20201019.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/VST_RF_pred_20201019.err
#SBATCH --mem 20000

# Vertex
## Ivacaftor
python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB08820
# ## Tezacaftor
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB11712
# ## Lumacaftor
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB09280
# ## Elexacaftor
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB15444
# ## Bamocaftor
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout VX-659

# # Galapgaos
# ## GLPG1837
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG1837
# ## GLPG2222
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout GLPG2222

# # Proteostatis
# ## PTI-428
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-428
# ## PTI-801
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-801
# ## PTI-808
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout PTI-808

# ## FDL169
# python predict/RF/RF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout FDL169