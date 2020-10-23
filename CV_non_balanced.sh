#!/bin/bash
#SBATCH --job-name=non_balanced
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/non_balanced_20201023.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/non_balanced_20201023.err
#SBATCH --mem 20000

# python cross_validation/make_folds/nested_cv_make_folds.py drugbank_v5.1.5 S0h

python cross_validation/kronSVM/nested_cv_make_kronSVM_clf.py drugbank_v5.1.5 S0h 

# python cross_validation/kronSVM/nested_cv_kronSVM_pred.py drugbank_v5.1.5 S0h