#!/bin/bash

##SBATCH --job-name=ondine_pred
##SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_pred_20200730.log
##SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/ondine_pred_20200730.err
##SBATCH --mem 20000

# ondine-oprhan
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h ondine-orphan DB12442 --center_norm
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h ondine-orphan DB00304 --center_norm 
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h ondine-orphan DB00294 --center_norm 

# sachant-tout
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h sachant-tout DB12442 --center_norm
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h sachant-tout DB00304 --center_norm 
PYTHONPATH=/Users/matthieu/ownCloud/Code/ python predict/kronSVM/kronSVM_pred_for_drug.py drugbank_v5.1.5 S0h sachant-tout DB00294 --center_norm 