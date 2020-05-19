import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

from process_dataset.DB_utils import ListInteractions 
from process_dataset.process_DB import get_DB
from make_K_inter import get_K_mol_K_prot
from make_K_train import InteractionsTrainDataset, make_K_train

from cv_get_folds import get_train_folds

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
                        help = "where or not to normalize the kernels, False \
                        by default")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name = args.process_name + '_' + args.DB_type
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + pattern_name + '/'

    #create directories
    if not os.path.exists(root + data_dir + '/' + 'CrossValidation'):
        os.mkdir(root + data_dir + '/' + 'CrossValidation')
        print("Cross Validation directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("Cross Validation directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    if not os.path.exists(root + data_dir + '/' + 'CrossValidation/kronSVM'):
        os.mkdir(root + data_dir + '/' + 'CrossValidation/kronSVM')
        print("kronSVM cross validation directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("kronSVM cross validation directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'CrossValidation/'
    kronsvm_cv_dirname = root + data_dir + 'CrossValidation/kronSVM/'

    C = 10.

    preprocessed_DB = get_DB(args.DB_version, args.DB_type, args.process_name)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.process_name,
                                 args.norm)

    # Get the train datasets
    train_folds = get_train_folds(args.DB_version, args.DB_type, args.process_name)
    # train_folds = pickle.load(open(cv_dirname + pattern_name + '_train_folds.data',
                            #   'rb'))
    nb_folds = len(train_folds)

    cv_list_clf = []
    cv_list_couples_of_clf, list_ind_false_inter = [], []

    for ifold in range(nb_folds):
        print("fold:", ifold)

        train_dataset = train_folds[ifold]
        true_inter = train_dataset.true_inter
        false_inter = train_dataset.false_inter

        K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
        y_train = np.concatenate((true_inter.interaction_bool, 
                                  false_inter.interaction_bool),
                                  axis=0)

        print("Training dataset's kernel of interactions prepared.")
        # We should add the list of forbidden

        # Create the classifier
        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_train, y_train)
        cv_list_clf.append(clf)

        # the list of couples in the train set are necessary to compute the 
        # similarity kernel for the interactions that we want to predict 
        # true_inter = train_set[2]
        # false_inter = train_set[3]
        list_couples = true_inter.list_couples + false_inter.list_couples
        cv_list_couples_of_clf.append(list_couples)
    
    print("Classifiers for the cross validation done.")

        # Optional
        # list_ind_false_inter.append(false_inter.ind_inter)
        
    # Classifier name
    if args.norm == True:
        cv_clf_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_clf_norm.data'
    else:
        cv_clf_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_clf.data'

    pickle.dump(cv_list_clf, 
                open(cv_clf_filename, 'wb'),
                protocol=2)

    # Couples of the classifier
    cv_couples_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_couples_of_clf.data'

    pickle.dump(cv_list_couples_of_clf, 
                open(cv_couples_filename, 'wb'), 
                protocol=2)
    
    print("Classifiers for the cross validation saved.")