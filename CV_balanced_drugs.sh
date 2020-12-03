#!/bin/bash
#SBATCH --job-name=balanced_drugs
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/balanced_drugs_20201027.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/balanced_drugs_20201027.err
#SBATCH --mem 20000

# python cross_validation/make_folds/nested_cv_make_folds.py drugbank_v5.1.5 S0h --balanced_on_drugs 

# python cross_validation/kronSVM/nested_cv_make_kronSVM_clf.py drugbank_v5.1.5 S0h --balanced_on_drugs

# python cross_validation/kronSVM/nested_cv_kronSVM_pred.py drugbank_v5.1.5 S0h 1 --balanced_on_drugs

python cross_validation/kronSVM/nested_cv_kronSVM_pred.py drugbank_v5.1.5 S0h 5 --balanced_on_drugs