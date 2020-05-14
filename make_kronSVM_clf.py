import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

from process_dataset.process_DB import get_DB
# from make_K_train import ListInteractions, get_list_couples_train, make_K_train
from make_K_train import ListInteractions, InteractionsTrainDataset, get_train_dataset, make_K_train
from make_K_inter import get_K_mol_K_prot

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
    if not os.path.exists(root + data_dir + '/' + 'Classifiers'):
        os.mkdir(root + data_dir + '/' + 'Classifiers')
        print("Classifiers directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("Classifiers directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    if not os.path.exists(root + data_dir + '/' + 'Classifiers/kronSVM'):
        os.mkdir(root + data_dir + '/' + 'Classifiers/kronSVM')
        print("kronSVM classifiers directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("kronSVM classifiers directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    clf_dirname = root + data_dir + 'Classifiers/kronSVM/'

    C = 10.

    preprocessed_DB = get_DB(args.DB_version, args.DB_type, args.process_name)

    # introduce part where we orphan etc ...

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.process_name,
                                 args.norm)

    list_seed = [71, 343, 928, 2027, 2]
    list_clf = []
    list_couples_of_clf, list_ind_false_inter = [], []

    for seed in list_seed:
        print("seed:", seed)

        # Create the train dataset
        train_dataset = get_train_dataset(seed, preprocessed_DB)
        # true_inter = train_dataset[0]
        # false_inter = train_dataset[1]
        true_inter = train_dataset.true_inter
        false_inter = train_dataset.false_inter

        # Compute the kernel of interactions
        # train_set = make_K_train(seed, preprocessed_DB, kernels)
        # K_train = train_set[0] 
        # y_train = train_set[1]

        K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
        y_train = np.concatenate((true_inter.interaction_bool, 
                                  false_inter.interaction_bool),
                                  axis=0)

        print("Training dataset's kernel of interactions prepared.")

        # Create the classifier
        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_train, y_train)
        list_clf.append(clf)

        # the list of couples in the train set are necessary to compute the 
        # similarity kernel for the interactions that we want to predict 
        # true_inter = train_set[2]
        # false_inter = train_set[3]
        list_couples = true_inter.list_couples + false_inter.list_couples
        list_couples_of_clf.append(list_couples)
    
    print("Classifiers done.")

        # Optional
        # list_ind_false_inter.append(false_inter.ind_inter)
        
    # Classifier name
    if args.norm == True:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_norm.data'
    else:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf.data'

    pickle.dump(list_clf, 
                open(clf_filename, 'wb'),
                protocol=2)

    # Couples of the classifier
    couples_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_couples_of_clf.data'

    pickle.dump(list_couples_of_clf, 
                open(couples_filename, 'wb'), 
                protocol=2)
    
    print("Classifiers saved.")