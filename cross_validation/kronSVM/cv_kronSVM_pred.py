import argparse 
import pickle
import numpy as np
import os

from sklearn.svm import SVC

from process_dataset.DB_utils import ListInteractions
from process_dataset.process_DB import get_DB
from make_K_inter import get_K_mol_K_prot
from cv_make_K_test import make_K_test
from cv_get_folds import get_test_folds

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions "

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
                        help = "where or not to normalize the kernels")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name = args.process_name + '_' + args.DB_type
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + pattern_name + '/'

    cv_dirname = root + data_dir + 'CrossValidation/'

    #create prediction output directory
    if not os.path.exists(root + data_dir + '/' + 'CrossValidation/kronSVM'):
        os.mkdir(root + data_dir + '/' + 'CrossValidation/kronSVM')
        print("kronSVM cross validation directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("kronSVM cross validation directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    kronsvm_cv_dirname = root + data_dir + 'CrossValidation/kronSVM/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.norm)

    # Get the test datasets
    test_folds = get_test_folds(args.DB_version, args.DB_type, args.process_name)
    # test_folds = pickle.load(open(cv_dirname + pattern_name + '_test_folds.data',
                            #   'rb'))
    nb_folds = len(test_folds)

    # get the classifiers
    if args.norm == True:
        cv_clf_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_clf_norm.data'
        output_filename = kronsvm_cv_dirname + pattern_name + '_kronSVM_cv_test_pred_norm.data'
    else:
        cv_clf_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_clf.data'
        output_filename = kronsvm_cv_dirname + pattern_name + '_kronSVM_cv_test_pred.data'
    cv_list_clf = pickle.load(open(cv_clf_filename, 'rb'))

    cv_couples_filename = kronsvm_cv_dirname + pattern_name + \
        '_kronSVM_cv_list_couples_of_clf.data'
    cv_list_couples_of_clf = pickle.load(open(cv_couples_filename, 'rb'))

    cv_pred = []

    for ifold in range(nb_folds):
        K_test = make_K_test(cv_list_couples_of_clf[ifold], 
                            test_folds[ifold].list_couples,
                            preprocessed_DB,
                            kernels)
        predict = cv_list_clf[ifold].predict_proba(K_test)[:,1]
        cv_pred.append(predict)
        print("Prediction for fold", ifold, "done.") 

    pickle.dump(cv_pred, open(output_filename, 'wb'))
    print("Cross validation done and saved.")