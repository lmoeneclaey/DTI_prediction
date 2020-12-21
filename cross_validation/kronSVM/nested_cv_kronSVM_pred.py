import argparse 
import copy
import pickle
import numpy as np
import os

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

# from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset
from DTI_prediction.cross_validation.kronSVM.cv_make_K_test import make_K_test
# from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_test_folds, get_train_folds

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

    parser.add_argument("--balanced_on_proteins", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--balanced_on_drugs", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

    # parser.add_argument("C", type = int,
    #                     help = "C hyperparameter in SVM algorithm")

    # parser.add_argument("--norm", default = False, action="store_true", 
    #                     help = "where or not to normalize the kernels")

    # parser.add_argument("--center_norm", default = False, action="store_true", 
    #                     help = "whether or not to center AND normalize the \
    #                         kernels, False by default")

    args = parser.parse_args()

    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'
    cv_dirname = root + data_dir + 'cross_validation/'
    kronsvm_cv_dirname = root + data_dir + 'cross_validation/kronSVM/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type)

    # Get the nested folds and classifiers
    nested_cv_dirname = root + data_dir + 'cross_validation/nested_folds/'
    if args.balanced_on_proteins == True:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname + args.DB_type \
            + '_nested_folds_double_balanced_' + str(args.nb_clf) \
            + '_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_double_balanced_' + str(args.nb_clf) \
            + '_clf_clf.data'
            output_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_double_balanced_' + str(args.nb_clf) \
            + '_clf_pred.data'
        else:
            nested_folds_array_filename = nested_cv_dirname + args.DB_type \
            + '_nested_folds_balanced_on_proteins_' + str(args.nb_clf) \
            + '_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_balanced_on_proteins_' + str(args.nb_clf) \
            + '_clf_clf.data'
            output_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_balanced_on_proteins_' + str(args.nb_clf) \
            + '_clf_pred.data'
    else:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname + args.DB_type \
            + '_nested_folds_balanced_on_drugs_' + str(args.nb_clf) \
            + '_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_balanced_on_drugs_' + str(args.nb_clf) \
            + '_clf_clf.data'
            output_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_balanced_on_drugs_' + str(args.nb_clf) \
            + '_clf_pred.data'
        else:
            nested_folds_array_filename = nested_cv_dirname + args.DB_type \
            + '_nested_folds_non_balanced_' + str(args.nb_clf) \
            + '_clf_array.data' 
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_non_balanced_' + str(args.nb_clf) \
            + '_clf_clf.data'
            output_filename = kronsvm_cv_dirname + args.DB_type \
            + '_kronSVM_cv_nested_non_balanced_' + str(args.nb_clf) \
            + '_clf_pred.data'

    nested_folds_array = pickle.load(open(nested_folds_array_filename, 'rb'))
    list_clf = pickle.load(open(cv_clf_filename, 'rb'))

    # # Get the test folds
    # test_folds_filename = nested_cv_dirname + args.DB_type \
    # + '_nested_folds_double_balanced_' + str(args.nb_clf) \
    # + '_clf_array.data'

    # test_folds_array = pickle.load(open(test_folds_filename, 'rb'))

    nb_clf = args.nb_clf

    if nb_clf==1:
        nb_folds = len(nested_folds_array)

        list_folds = []
        cv_list_clf = []
        test_folds = []
        for ifold in range(len(nested_folds_array)):
            # train folds
            fold_dataset = get_couples_from_array(nested_folds_array[ifold])
            list_folds.append([fold_dataset])

            # classifier
            cv_list_clf.append([list_clf[ifold]])

            # test folds
            test_fold_dataset = get_couples_from_array(nested_folds_array[ifold])
            test_folds.append(test_fold_dataset)

    else:
        nb_folds = len(nested_folds_array[0])

        # train folds
        list_folds = []
        for iclf in range(nb_clf):
            list_folds_per_clf = []
            for ifold in range(nb_folds):
                fold_dataset = get_couples_from_array(nested_folds_array[iclf][ifold])
                list_folds_per_clf.append(fold_dataset)
            list_folds.append(list_folds_per_clf)

        # classifier
        cv_list_clf = copy.deepcopy(list_clf)

        # test folds
        test_folds = []
        for ifold in range(nb_folds):
            test_fold_dataset = get_couples_from_array(nested_folds_array[0][ifold])
            # test_fold_dataset = get_couples_from_array(nested_folds_array[0][(ifold+3)%5])
            test_folds.append(test_fold_dataset)

    cv_list_pred = []

    for ifold in range(nb_folds):

        print("fold:", ifold)
        test_dataset = test_folds[ifold]

        cv_list_pred_per_fold = []
        for iclf in range(nb_clf):

            train_folds = [list_folds[iclf][(ifold+1)%5],
                           list_folds[iclf][(ifold+2)%5],
                           list_folds[iclf][(ifold+3)%5],
                           list_folds[iclf][(ifold+4)%5]]

            # train_folds = [list_folds[iclf][(ifold+1)%5],
            #                list_folds[iclf][(ifold+2)%5],
            #                list_folds[iclf][(ifold+4)%5],
            #                list_folds[iclf][(ifold+5)%5]]

            train_dataset = sum(train_folds)

            K_test = make_K_test(list_couples_train = train_dataset.list_couples, 
                                list_couples_test = test_dataset.list_couples,
                                preprocessed_DB = preprocessed_DB,
                                kernels = kernels)

            clf = cv_list_clf[ifold][iclf]
            cv_list_pred_per_fold.append(clf.predict_proba(K_test)[:,1])
        cv_list_pred.append(cv_list_pred_per_fold)

        print("Predictions for fold", ifold, "done.") 

    pickle.dump(cv_list_pred, open(output_filename, 'wb'))
    print("Cross validation done and saved.")