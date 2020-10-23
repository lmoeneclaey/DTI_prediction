import argparse
import csv
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
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array

from sklearn import preprocessing

from DTI_prediction.make_classifiers.FNN_clf.FNN_utils import FNN_model

root = '../CFTR_PROJECT/'

def get_Xcouple(x, X_mol, X_prot, dict_mol2ind, dict_prot2ind):
    X = np.zeros((len(x), X_mol.shape[1] + X_prot.shape[1]))
    for i in range(len(x)):
        prot, mol = x[i]
        X[i, :] = np.concatenate([X_mol[dict_mol2ind[mol], :],
                                  X_prot[dict_prot2ind[prot], :]])
    return X

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Process the kernel of interactions of a list of molecules and proteins \
        and create the corresponding kronSVM classifier.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name =  args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name

    #create directories
    if not os.path.exists(root + 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name):
        os.mkdir(root + 'data/' + args.DB_version + '/' + args.DB_type + '/' +  pattern_name)
        print("Directory", pattern_name, "for",  args.DB_version, "created")
    else: 
        print("Directory", pattern_name, "for",  args.DB_version, " already exists")

    if not os.path.exists(root + data_dir + '/' + 'classifiers'):
        os.mkdir(root + data_dir + '/' + 'classifiers')
        print("Classifiers directory for", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("Classifiers directory for", pattern_name, ",", args.DB_version,
        "already exists.")

    if not os.path.exists(root + data_dir + '/' + 'classifiers/FNN'):
        os.mkdir(root + data_dir + '/' + 'classifiers/FNN')
        print("FNN classifiers directory for ", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("FNN classifiers directory for ", pattern_name, ",", args.DB_version,
        "already exists.")

    # # Get the train datasets 

    # train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'
    # train_datasets_array_filename = train_datasets_dirname + pattern_name + \
    #     '_train_datasets_array.data'

    # train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    # nb_clf = len(train_datasets_array)

    # list_train_datasets = []
    # for iclf in range(nb_clf):
    #     train_dataset = get_couples_from_array(train_datasets_array[iclf])
    #     list_train_datasets.append(train_dataset)

    # Get the nested folds
    nested_cv_dirname = root + 'data/' + args.DB_version + '/' + args.DB_type + '/' + 'cross_validation/nested_folds/'
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

    #####

    clf_dirname = root + data_dir + '/classifiers/FNN/'

    DB = get_DB(args.DB_version, args.DB_type)

    features = get_mol_prot_features(args.DB_version, args.DB_type)
    X_mol = features[0]
    X_prot = features[1]

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(X_prot)

    X_prot_scaled = standard_scaler.transform(X_prot)

    param = {'layers_units': [2000, 1000, 100], 
             'BN': True, # batchnorm
             'reg': 0.0, # no regularisation
             'drop': 0.0, # no drop out  
             'n_epochs': 100, # number of epochs 
             'init_lr': 0.001, # initial learning rate
             'patience': 20, 
             'lr_scheduler': {"name": "LearningRateScheduler", "rate": 0.9},
             'batch_size': 100}

    ml = FNN_model(param=param, DB=DB, X_prot=X_prot_scaled, X_mol=X_mol)
    ml.build()

    ifold = 1

    X_tr = get_Xcouple(sum(train_folds[ifold]).list_couples,
                   X_mol,
                   X_prot_scaled,
                   DB.drugs.dict_mol2ind,
                   DB.proteins.dict_prot2ind)

    y_tr = sum(train_folds[ifold]).interaction_bool.reshape(-1,)

    X_val = get_Xcouple(sum(test_folds[ifold]).list_couples,
                    X_mol,
                    X_prot_scaled,
                    DB.drugs.dict_mol2ind,
                    DB.proteins.dict_prot2ind)

    y_val = sum(test_folds[ifold]).interaction_bool.reshape(-1,)

    ml.fit(X_tr, y_tr, X_val, y_val)

    print("Classifiers done.")
        
    clf_filename = clf_dirname + pattern_name + '_FNN_test_clf.h5'

    ml.model.save(clf_filename)
    
    print("Classifiers saved.")
