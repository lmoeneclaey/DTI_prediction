import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset, make_K_train

from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_train_folds

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Process the kernel of interactions of a list of molecules and proteins \
        and create the corresponding kronSVM classifier for cross validation analysis.")

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
                        help = "where or not to normalize the kernels, False \
                        by default")

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

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.norm)

    # Get the train datasets
    train_folds = get_train_folds(args.DB_version, args.DB_type, args.nb_clf)

    nb_folds = len(train_folds)
    nb_clf = len(train_folds[0])

    cv_list_clf = []
    cv_list_couples_of_clf = []

    for ifold in range(nb_folds):
        print("fold:", ifold)

        cv_list_clf_per_fold = []
        cv_list_couples_of_clf_per_fold = []

        for iclf in range(nb_clf):

            train_dataset = train_folds[ifold][iclf]
            true_inter = train_dataset.true_inter
            false_inter = train_dataset.false_inter

            K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
            y_train = np.concatenate((true_inter.interaction_bool, 
                                      false_inter.interaction_bool),
                                      axis=0)

            # Create the classifier
            clf = SVC(C=args.C, 
                      kernel='precomputed', 
                      probability=True, 
                      class_weight='balanced')
            clf.fit(K_train, y_train.ravel())
            cv_list_clf_per_fold.append(clf)

            # list of couples
            list_couples = true_inter.list_couples + false_inter.list_couples
            cv_list_couples_of_clf_per_fold.append(list_couples)

        cv_list_clf.append(cv_list_clf_per_fold)
        cv_list_couples_of_clf.append(cv_list_couples_of_clf_per_fold)
    
    print("Classifiers for the cross validation done.")

        # Optional
        # list_ind_false_inter.append(false_inter.ind_inter)
        
    # Classifier name
    if args.norm == True:
        cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_list_clf_norm.data'
    else:
        cv_clf_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_list_clf.data'

    pickle.dump(cv_list_clf, 
                open(cv_clf_filename, 'wb'),
                protocol=2)

    # Couples of the classifier
    cv_couples_filename = kronsvm_cv_dirname + args.DB_type + \
        '_kronSVM_cv_list_couples_of_clf.data'

    pickle.dump(cv_list_couples_of_clf, 
                open(cv_couples_filename, 'wb'), 
                protocol=2)
    
    print("Classifiers for the cross validation saved.")