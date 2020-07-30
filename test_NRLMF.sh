#!/bin/bash
#SBATCH --job-name=test_NRLMF
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/test_NRLMF_20200728.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/test_NRLMF_20200728.err
#SBATCH --mem 20000

## Ivacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB08820 --center_norm