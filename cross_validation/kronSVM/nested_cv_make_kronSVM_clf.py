import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

# from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset, make_K_train
from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import make_K_train
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array

# from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_train_folds

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    print("debut du script")

    parser = argparse.ArgumentParser(
    "Process the kernel of interactions of a list of molecules and proteins \
        and create the corresponding kronSVM classifier for cross validation analysis.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    # parser.add_argument("nb_clf", type = int,
    #                     help = "number of classifiers for future predictions, example = 5")

    parser.add_argument("--balanced_on_proteins", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--balanced_on_drugs", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

    # parser.add_argument("C", type = int,
    #                     help = "C hyperparameter in SVM algorithm")

    # parser.add_argument("--norm", default = False, action="store_true", 
    #                     help = "where or not to normalize the kernels, False \
    #                     by default")

    # parser.add_argument("--center_norm", default = False, action="store_true", 
    #                     help = "whether or not to center AND normalize the \
    #                         kernels, False by default")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    if not os.path.exists(root + data_dir + '/' + 'cross_validation/kronSVM'):
        os.mkdir(root + data_dir + '/' + 'cross_validation/kronSVM')
        print("kronSVM cross validation directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("kronSVM cross validation directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'cross_validation/'
    kronsvm_cv_dirname = root + data_dir + 'cross_validation/kronSVM/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, center_norm = True, norm = True)

    # Get the nested folds
    nested_cv_dirname = root + data_dir + 'cross_validation/nested_folds/'

    if args.balanced_on_proteins == True:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_double_balanced_5_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
                    '_kronSVM_cv_nested_double_balanced_clf_5_clf.data'
        else:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_balanced_on_proteins_5_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
                    '_kronSVM_cv_nested_balanced_on_proteins_clf_5_clf.data'
    else:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_balanced_on_drugs_5_clf_array.data'
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
                    '_kronSVM_cv_nested_balanced_on_drugs_clf_5_clf.data'
        else:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_non_balanced_5_clf_array.data' 
            cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
                    '_kronSVM_cv_nested_non_balanced_clf_5_clf.data'

    nested_folds_array = pickle.load(open(nested_folds_array_filename, 'rb'))

    nb_clf = len(nested_folds_array)
    nb_folds = len(nested_folds_array[0])

    list_folds = []
    for iclf in range(nb_clf):
        list_folds_per_clf = []
        for ifold in range(len(nested_folds_array)):
            fold_dataset = get_couples_from_array(nested_folds_array[iclf][ifold])
            list_folds_per_clf.append(fold_dataset)
        list_folds.append(list_folds_per_clf)

    cv_list_clf = []

    for ifold in range(nb_folds):
        
        print("fold:", ifold)

        cv_list_clf_per_fold = []
        for iclf in range(nb_clf):

            train_folds = [list_folds[iclf][(ifold+1)%5],
                           list_folds[iclf][(ifold+2)%5],
                           list_folds[iclf][(ifold+4)%5],
                           list_folds[iclf][(ifold+5)%5]]

            train_dataset = sum(train_folds)

            K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
            y_train = train_dataset.interaction_bool

            clf = SVC(C=10, 
                    kernel='precomputed', 
                    probability=True, 
                    class_weight='balanced')
            clf.fit(K_train, y_train.ravel())
            cv_list_clf_per_fold.append(clf)

        cv_list_clf.append(cv_list_clf_per_fold)
    
    print("Classifiers for the cross validation done.")

    pickle.dump(cv_list_clf, 
                open(cv_clf_filename, 'wb'),
                protocol=2)
    
    print("Classifiers for the cross validation saved.")