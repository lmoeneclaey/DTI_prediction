import argparse
import csv
import collections
import copy
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys

from DTI_prediction.process_dataset.process_DB import get_DB

from DTI_prediction.make_kernels.get_kernels import get_mol_prot_features
from DTI_prediction.make_classifiers.RF_clf.make_RF_clf import get_Xcouple

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array
from DTI_prediction.utils.performance_utils import get_clf_perf
from DTI_prediction.utils.train_dataset_utils import get_number_of_interactions_per_mol

root = '../CFTR_PROJECT/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    "Cross validation analysis of the Random Forest algorithm.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    if not os.path.exists(root + data_dir + '/' + 'cross_validation/RF'):
        os.mkdir(root + data_dir + '/' + 'cross_validation/RF')
        print("RF cross validation directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("RF cross validation directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'cross_validation/'
    rf_cv_dirname = root + data_dir + 'cross_validation/RF/'

    DB = get_DB(args.DB_version, args.DB_type)

    # Get the features
    features = get_mol_prot_features(args.DB_version, args.DB_type)
    X_mol = features[0]
    X_prot = features[1]

    # Get the nested folds
    nested_cv_dirname = root + data_dir + 'cross_validation/nested_folds/'
    nested_folds_array_filename = nested_cv_dirname + args.DB_type + '_nested_folds_array.data'

    nested_folds_array = pickle.load(open(nested_folds_array_filename, 'rb'))

    nb_folds = len(nested_folds_array)

    list_folds = []
    for ifold in range(nb_folds):
        fold_dataset = get_couples_from_array(nested_folds_array[ifold])
        list_folds.append(fold_dataset)

    test_folds = [[list_folds[0],list_folds[1]], 
                  [list_folds[2],list_folds[3]], 
                  [list_folds[4],list_folds[5]]]
    train_folds = [[list_folds[2],list_folds[3],list_folds[4],list_folds[5]],
                   [list_folds[0],list_folds[1],list_folds[4],list_folds[5]],
                   [list_folds[0],list_folds[1],list_folds[2],list_folds[3]]] 

    # For the performance per category
    dict_cat = {0:'0', 1:'[1,4]', 2:'[5,10]', 3:'> 10'}

    # CV outer loop
    nb_folds = 3

    pred_perf = []
    pred_perf_per_fold = []

    for iouter in range(nb_folds):

        print("CV - outer", iouter, "\n")
    
        n_estimators_list = [200,400,600]
        min_samples_leaf_list = [1,2,5,10]
        min_samples_split_list = [2,5]
        max_depth_list = [10,20]

        # max_prec = 0
        n_estimators_opt = 600
        min_samples_leaf_opt = 1
        min_samples_split_opt = 5
        max_depth_opt = 20

        # for n_estimators in n_estimators_list:
        #     for min_samples_leaf in min_samples_leaf_list:
        #         for min_samples_split in min_samples_split_list:
        #             for max_depth in max_depth_list:

        #                 print("Parameter setting:")
        #                 print("\t - the number of estimators=", n_estimators)
        #                 print("\t - the minimum number of samples required to be at a leaf node=", min_samples_leaf)
        #                 print("\t - the minimum number of samples required to split an internal node=", min_samples_split)
        #                 print("\t - the maximum depth of the tree=", max_depth)

        #                 rf = RandomForestClassifier(n_estimators=n_estimators,
        #                                             min_samples_leaf=min_samples_leaf,
        #                                             min_samples_split=min_samples_split,
        #                                             max_depth=max_depth, 
        #                                             random_state=53)

        #                 pred_perf = []
        #                 pred_perf_per_fold = []

        #                 for ifold in range(4):

        #                     print("CV - inner", ifold, "\n")

        #                     # Test inner
    
        #                     test_inner_fold = train_folds[iouter][ifold]
        #                     test_inner_pd = pd.DataFrame(test_inner_fold.array,
        #                                                   columns= ['UniProt ID', 
        #                                                             'DrugBank ID', 
        #                                                             'interaction_bool'])
        #                     test_inner_pd["interaction_bool"] = pd.to_numeric(test_inner_pd["interaction_bool"])

        #                     X_te = get_Xcouple(test_inner_fold.list_couples,
        #                                        X_mol,
        #                                        X_prot,
        #                                        DB.drugs.dict_mol2ind,
        #                                        DB.proteins.dict_prot2ind)
        #                     y_te = test_inner_fold.interaction_bool.reshape(-1,)
    
        #                     # Train inner

        #                     train_inner_folds = copy.deepcopy(train_folds[iouter])
        #                     train_inner_folds.pop(ifold-4)
                            
        #                     train_inner_fold  = sum(train_inner_folds)
        #                     train_inner_pd = pd.DataFrame(train_inner_fold.array,
        #                                                   columns= ['UniProt ID', 
        #                                                             'DrugBank ID', 
        #                                                             'interaction_bool'])
        #                     train_inner_pd["interaction_bool"] = pd.to_numeric(train_inner_pd["interaction_bool"])
    
        #                     X_tr = get_Xcouple(train_inner_fold.list_couples,
        #                                        X_mol,
        #                                        X_prot,
        #                                        DB.drugs.dict_mol2ind,
        #                                        DB.proteins.dict_prot2ind)
        #                     y_tr = train_inner_fold.interaction_bool.reshape(-1,)

        #                     # Fit
    
        #                     rf.fit(X_tr, y_tr)

        #                     # Predict
    
        #                     pred = rf.predict(X_te)

        #                     # Post-process

        #                     result = get_number_of_interactions_per_mol(train_dataset_pd=train_inner_pd, 
        #                                                                 test_dataset_pd=test_inner_pd)
        #                     result['pred'] = pred

        #                     # Performance general and per category

        #                     pred_perf.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result['pred'], y_true=result['interaction_bool'])).iloc[0])
                        
        #                     pred_perf_per_cat_fold = []
        #                     for icat in range(4):
        #                         result_per_cat = result[result['category']==dict_cat[icat]]


        #                         pred_perf_per_cat_fold.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result_per_cat['pred'], y_true=result_per_cat['interaction_bool'])).iloc[0])
        #                     pred_perf_per_fold.append(pred_perf_per_cat_fold)

        #                 all_inner_folds_perf = pd.concat(pred_perf, axis=1).round(2)
        #                 final_perf = pd.DataFrame({'mean':all_inner_folds_perf.mean(axis=1).round(2), 
        #                                             'var':all_inner_folds_perf.var(axis=1).round(2)})

        #                 for icat in range(4):
                        
        #                     pred_perf_per_cat = []
        #                     for ifold in range(4):
        #                         pred_perf_per_cat.append(pred_perf_per_fold[ifold][icat])
                        
        #                     pred_perf_per_cat_pd = pd.concat(pred_perf_per_cat, axis=1).round(2)
                        
        #                     final_perf['mean'+dict_cat[icat]]=pred_perf_per_cat_pd.mean(axis=1).round(2)
        #                     final_perf['var'+dict_cat[icat]]=pred_perf_per_cat_pd.var(axis=1).round(2)

        #                 print(final_perf)

        #                 if final_perf['mean'].iloc[4] > max_prec:
        #                     mac_prec = final_perf['mean'].iloc[4]
        #                     n_estimators_opt = n_estimators
        #                     min_samples_leaf_opt = min_samples_leaf
        #                     min_samples_split_opt = min_samples_split
        #                     max_depth_opt = max_depth
        #                     pref_opt = final_perf

        # # Out of the loop on parameters
        # cmd = "-----------------------------------------------"
        # print("Optimal parameter setting:\n")
        # print("\t - the optimal number of estimators=", n_estimators_opt)
        # print("\t - the optimal minimum number of samples required to be at a leaf node=", min_samples_leaf_opt)
        # print("\t - the optimal minimum number of samples required to split an internal node=", min_samples_split_opt)
        # print("\t - the optimal maximum depth of the tree=", max_depth_opt)

        # print("Mean performance on validation set:")
        # print(pref_opt)
        # print("\n")

        # Test outer

        X_test_outer = sum(test_folds[iouter])

        test_outer_pd = pd.DataFrame(X_test_outer.array, 
                                      columns=['UniProt ID', 
                                               'DrugBank ID', 
                                               'interaction_bool'])
        test_outer_pd["interaction_bool"] = pd.to_numeric(test_outer_pd["interaction_bool"])
        
        X_te = get_Xcouple(X_test_outer.list_couples,
                           X_mol,
                           X_prot,
                           DB.drugs.dict_mol2ind,
                           DB.proteins.dict_prot2ind)

        # Train outer

        X_train_outer = sum(train_folds[iouter])

        train_outer_pd = pd.DataFrame(X_train_outer.array, 
                                      columns=['UniProt ID', 
                                               'DrugBank ID', 
                                               'interaction_bool'])
        train_outer_pd["interaction_bool"] = pd.to_numeric(train_outer_pd["interaction_bool"])

        X_tr = get_Xcouple(X_train_outer.list_couples,
                           X_mol,
                           X_prot,
                           DB.drugs.dict_mol2ind,
                           DB.proteins.dict_prot2ind)

        y_tr = X_train_outer.interaction_bool.reshape(-1,)

        rf_opt = RandomForestClassifier(n_estimators=n_estimators_opt,
                                        min_samples_leaf=min_samples_leaf_opt,
                                        min_samples_split=min_samples_split_opt,
                                        max_depth=max_depth_opt)

        # Fit

        rf_opt.fit(X_tr, y_tr)

        # Predict

        pred = rf_opt.predict(X_te)

        # Post-process
        result = get_number_of_interactions_per_mol(train_dataset_pd=train_outer_pd, 
                                                    test_dataset_pd=test_outer_pd)
        result['pred'] = pred

        pred_perf.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result['pred'], y_true=result['interaction_bool'].astype(int))).iloc[0])

        perf_per_cat_outer = []
        perf_per_cat_outer.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result['pred'], 
                                                                      y_true=result['interaction_bool'].astype(int))).iloc[0])
        pred_perf_per_cat_fold = []
        for icat in range(4):
            result_per_cat = result[result['category']==dict_cat[icat]]
            perf_per_cat_outer.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result_per_cat['pred'], y_true=result_per_cat['interaction_bool'].astype(int))).iloc[0])
            pred_perf_per_cat_fold.append(pd.DataFrame.from_dict(get_clf_perf(y_pred=result_per_cat['pred'], y_true=result_per_cat['interaction_bool'].astype(int))).iloc[0])
        pred_perf_per_fold.append(pred_perf_per_cat_fold)

        perf_per_cat_outer_pd = pd.concat(perf_per_cat_outer, axis=1).round(2)

        print("Evaluation on test")
        print(perf_per_cat_outer_pd)

        print("----------------------------------------")

    all_folds_perf = pd.concat(pred_perf, axis=1).round(2)
    final_perf = pd.DataFrame({'mean':all_folds_perf.mean(axis=1).round(2), 
                                'var':all_folds_perf.var(axis=1).round(2)})

    for icat in range(4):
    
        pred_perf_per_cat = []
        for ifold in range(3):
            pred_perf_per_cat.append(pred_perf_per_fold[ifold][icat])
    
        pred_perf_per_cat_pd = pd.concat(pred_perf_per_cat, axis=1).round(2)
    
        final_perf['mean'+dict_cat[icat]]=pred_perf_per_cat_pd.mean(axis=1).round(2)
        final_perf['var'+dict_cat[icat]]=pred_perf_per_cat_pd.var(axis=1).round(2)

    print(final_perf)

    pickle.dump(final_perf, open(rf_cv_dirname + args.DB_type + \
        '_rf_nested_perf_optimal_parameters.data', 'wb'))