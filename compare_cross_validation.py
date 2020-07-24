import numpy as np
import os
import pandas as pd
import pickle
import sys
import seaborn as sns
import tensorflow as tf

from sklearn import metrics
import sklearn.model_selection as model_selection

def sround(score):
    return round(score * 100, 2)

sys.path.append('..')

nb_ligands_kernel_dir = "../../Thèse/Chemogenomics/Matthieu/Nb_ligands_kernels/"

root = '../CFTR_PROJECT/'

DB_version = "drugbank_v5.1.5"
DB_type = "S0h"

# data_dir variable 
data_dir = 'data/' + DB_version + '/' + DB_type

from DTI_prediction.process_dataset.process_DB import get_DB
DB = get_DB(DB_version, DB_type)

cv_dirname = root + data_dir + '/cross_validation/'
kronsvm_cv_dirname = root + data_dir + '/cross_validation/kronSVM/'

# 1 - Get the test folds 

from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_test_folds
from performance_utils import get_clf_perf, precision

test_folds = get_test_folds(DB_version, DB_type)
nb_folds = len(test_folds)

nb_predictions = 0
for ifold in range(nb_folds):
    nb_predictions = nb_predictions + test_folds[ifold].nb

total_true_list = []
for ifold in range(nb_folds):
    total_true_list = total_true_list + test_folds[ifold].interaction_bool.reshape(-1,).tolist()
total_true = np.array(total_true_list)

# A - Aucun traitement

print("A - Les kernels ne sont pas traités.")


raw_predictions_filename = kronsvm_cv_dirname + "S0h_kronSVM_cv_C_10_5_pred_20200624.data"
raw_predictions = pickle.load(open(raw_predictions_filename, 'rb'))

nb_all_clf = len(raw_predictions[0])

predictions_list = []
for ifold in range(nb_folds):   
    predictions_per_fold = np.zeros((len(raw_predictions[ifold][0]),nb_all_clf))
    for iclf in range(nb_all_clf):
        predictions_per_fold_per_clf = raw_predictions[ifold][iclf]
        predictions_per_fold[:,iclf] = predictions_per_fold_per_clf
        
    predictions_list.append(predictions_per_fold)

predictions = np.concatenate(predictions_list, axis=0)

predictions_pd = pd.DataFrame(predictions)

predictions_cv = pd.DataFrame({'true' : total_true,
                               'moy': np.average(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'max': np.max(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'min': np.min(predictions_pd[list(predictions_pd.columns)], axis=1)})


print("Moyenne")
pred = round(predictions_cv['moy'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['moy'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))


print("Maximum")
pred = round(predictions_cv['max'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['max'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

print("Minimum")
pred = round(predictions_cv['min'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['min'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

# B - Kprot est normé

print("B - Kprot est normé")

raw_predictions_filename = kronsvm_cv_dirname + "S0h_kronSVM_cv_C_10_5_pred_Kprot_norm_20200708.data"
raw_predictions = pickle.load(open(raw_predictions_filename, 'rb'))

nb_all_clf = len(raw_predictions[0])

predictions_list = []
for ifold in range(nb_folds):   
    predictions_per_fold = np.zeros((len(raw_predictions[ifold][0]),nb_all_clf))
    for iclf in range(nb_all_clf):
        predictions_per_fold_per_clf = raw_predictions[ifold][iclf]
        predictions_per_fold[:,iclf] = predictions_per_fold_per_clf
        
    predictions_list.append(predictions_per_fold)

predictions = np.concatenate(predictions_list, axis=0)

predictions_pd = pd.DataFrame(predictions)

predictions_cv = pd.DataFrame({'true' : total_true,
                               'moy': np.average(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'max': np.max(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'min': np.min(predictions_pd[list(predictions_pd.columns)], axis=1)})

print("Moyenne")
pred = round(predictions_cv['moy'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['moy'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

print("Maximum")
pred = round(predictions_cv['max'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['max'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

print("Minimum")
pred = round(predictions_cv['min'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['min'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

# C - Les kernels sont centrés normés

print("C - Les kernels sont centrés normés")

raw_predictions_filename = kronsvm_cv_dirname + "S0h_kronSVM_cv_C_10_5_pred_tout_centre_norme_20200624.data"
raw_predictions = pickle.load(open(raw_predictions_filename, 'rb'))

nb_all_clf = len(raw_predictions[0])

predictions_list = []
for ifold in range(nb_folds):   
    predictions_per_fold = np.zeros((len(raw_predictions[ifold][0]),nb_all_clf))
    for iclf in range(nb_all_clf):
        predictions_per_fold_per_clf = raw_predictions[ifold][iclf]
        predictions_per_fold[:,iclf] = predictions_per_fold_per_clf
        
    predictions_list.append(predictions_per_fold)

predictions = np.concatenate(predictions_list, axis=0)

predictions_pd = pd.DataFrame(predictions)

predictions_cv = pd.DataFrame({'true' : total_true,
                               'moy': np.average(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'max': np.max(predictions_pd[list(predictions_pd.columns)], axis=1),
                               'min': np.min(predictions_pd[list(predictions_pd.columns)], axis=1)})

print("Moyenne")
pred = round(predictions_cv['moy'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['moy'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

print("Maximum")
pred = round(predictions_cv['max'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['max'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))

print("Minimum")
pred = round(predictions_cv['min'])
true = predictions_cv['true']
print(get_clf_perf(y_pred=predictions_cv['min'], y_true=true))
print(metrics.confusion_matrix(y_pred=pred, y_true=true))
print("False Positive Rate:", 1 - metrics.precision_score(y_pred=pred, y_true=true))