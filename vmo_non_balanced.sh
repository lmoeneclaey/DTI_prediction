#!/bin/bash
#SBATCH --job-name=VMO_non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/vmo_non_balanced_20201127.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/vmo_non_balanced_20201127.err
#SBATCH --mem 20000

# train dataset
python make_classifiers/make_non_balanced_train_dataset.py drugbank_v5.1.5 S0h vmo_non_balanced --orphan DB08820 --orphan DB11712 --orphan DB09280 --orphan DB15444

# classifier
python make_classifiers/kronSVM_clf/make_kronSVM_clf.py drugbank_v5.1.5 S0h vmo_non_balanced 

# predictions
# Vertex
## Ivacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced DB08820 
## Tezacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced DB11712 
## Lumacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced DB09280 
## Elexacaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced DB15444 
## Bamocaftor
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced VX-659 

# Galapgaos
## GLPG1837
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced GLPG1837 
## GLPG2222
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced GLPG2222 

# Proteostatis
## PTI-428
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced PTI-428 
## PTI-801
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced PTI-801 
## PTI-808
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced PTI-808 

# FDL169
python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h vmo_non_balanced FDL169 
