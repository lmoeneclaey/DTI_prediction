#!/bin/bash
#SBATCH --job-name=CV_rf_test
#SBATCH --output=/mnt/data4/mnajm/CFTR_PROJECT/log/CV_rf_test_20201013.log
#SBATCH --error=/mnt/data4/mnajm/CFTR_PROJECT/log/CV_rf_test_20201013.err
#SBATCH --mem 20000

python -c 'from DTI_prediction.utils.package_utils import get_clf_perf'