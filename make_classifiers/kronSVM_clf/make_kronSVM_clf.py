import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import make_K_train

root = './../CFTR_PROJECT/'

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

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--center_norm", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

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

    if not os.path.exists(root + data_dir + '/' + 'classifiers/kronSVM'):
        os.mkdir(root + data_dir + '/' + 'classifiers/kronSVM')
        print("kronSVM classifiers directory for ", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("kronSVM classifiers directory for ", pattern_name, ",", args.DB_version,
        "already exists.")

    # Get the train datasets 

    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'
    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'

    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    nb_clf = len(train_datasets_array)

    list_train_datasets = []
    for iclf in range(nb_clf):
        train_dataset = get_couples_from_array(train_datasets_array[iclf])
        list_train_datasets.append(train_dataset)

    clf_dirname = root + data_dir + '/classifiers/kronSVM/'

    C = 10.

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.center_norm, args.norm)

    list_clf = []

    for iclf in range(nb_clf):

        train_dataset = list_train_datasets[iclf]

        # Compute the kernel of interactions
        K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
        y_train = train_dataset.interaction_bool

        print("Training dataset's kernel of interactions prepared.")

        # Create the classifier
        clf = SVC(C=C, 
                  kernel='precomputed', 
                  probability=True, 
                  class_weight='balanced')
        clf.fit(K_train, y_train.ravel())
        list_clf.append(clf)
    
    print("Classifiers done.")
        
    # Classifier name
    if args.center_norm == True:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_centered_norm.data'
    elif args.norm == True:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_norm.data'
    else:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf.data'

    pickle.dump(list_clf, 
                open(clf_filename, 'wb'),
                protocol=2)
    
    print("Classifiers saved.")