import argparse 
import pickle
import numpy as np
import os

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset
from DTI_prediction.cross_validation.kronSVM.cv_make_K_test import make_K_test
from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_test_folds

from progress.bar import IncrementalBar

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions in cross_validation "

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("nb_clf", type = int,
                        help = "number of classifiers for future predictions, example = 5")

    parser.add_argument("C", type = int,
                        help = "C hyperparameter in SVM algorithm")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "where or not to normalize the kernels")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    cv_dirname = root + data_dir + 'cross_validation/'
    kronsvm_cv_dirname = root + data_dir + 'cross_validation/kronSVM/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.norm)

    # Get the test datasets
    test_folds = get_test_folds(args.DB_version, args.DB_type)
    nb_folds = len(test_folds)

    # get the classifiers
    if args.norm == True:
        cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_C_' + str(args.C) + '_' + str(args.nb_clf) + '_clf_norm_identity.data'
        output_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_C_' + str(str(args.C)) + '_' + str(args.nb_clf) + '_pred_norm_identity.data'
    else:
        cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_C_' + str(str(args.C)) + '_' + str(args.nb_clf) + '_clf_identity.data'
        output_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_C_' + str(args.C) + '_' + str(args.nb_clf) + '_pred_identity.data'

    cv_list_clf = pickle.load(open(cv_clf_filename, 'rb'))

    cv_couples_filename = kronsvm_cv_dirname + args.DB_type + \
    '_kronSVM_cv_C_' + str(args.C) + '_' + str(args.nb_clf) + '_clf_couples_identity.data'
    cv_list_couples_of_clf = pickle.load(open(cv_couples_filename, 'rb'))

    cv_pred = []

    bar = IncrementalBar('Progression', max = nb_folds*args.nb_clf)

    for ifold in range(nb_folds):
        print(ifold)

        pred = []
        for iclf in range(args.nb_clf):
            print(iclf)
            K_test = make_K_test(cv_list_couples_of_clf[ifold][iclf], 
                                test_folds[ifold].list_couples,
                                preprocessed_DB,
                                kernels)
            clf = cv_list_clf[ifold][iclf]
            pred.append(clf.predict_proba(K_test)[:,1])
            bar.next()
        cv_pred.append(pred)
        print("Prediction for fold", ifold, "done.") 

    pickle.dump(cv_pred, open(output_filename, 'wb'))
    print("Cross validation done and saved.")