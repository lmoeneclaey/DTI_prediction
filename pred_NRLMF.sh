#!/bin/bash
#SBATCH --job-name=pred_NRLMF
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/pred_NRLMF_20200901.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/pred_NRLMF_20200901.err
#SBATCH --mem 20000

## Ivacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB08820 --center_norm

## Lumacaftor
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h vertex-sachant-tout DB09280 --center_norm

## Alvespimycin
python predict/NRLMF/NRLMF_pred_for_drug.py drugbank_v5.1.5 S0h sachant-tout DB12442 --center_norm