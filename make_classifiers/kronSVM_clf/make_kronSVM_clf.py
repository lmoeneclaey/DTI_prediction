import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.svm import SVC

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.process_dataset.correct_interactions import get_orphan, correct_interactions
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset, get_train_dataset, make_K_train

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

    parser.add_argument("--orphan", type = str, action='append',
                        help = "molecules which you want to orphanize in the \
                            train data set")

    parser.add_argument("--correct_interactions", default = False, action="store_true",
                        help = "whether or not to add or correct some \
                            interactions, False by default")

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

    clf_dirname = root + data_dir + '/classifiers/kronSVM/'

    C = 10.

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)


    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.norm)

    list_seed = [2821, 1148, 1588, 188, 933]
    list_clf = []
    list_couples_of_clf = []

    corrected_DB = copy.deepcopy(preprocessed_DB)
    
    for dbid in args.orphan:
        corrected_DB = get_orphan(DB=corrected_DB, dbid=dbid)

    if args.correct_interactions == True:

        corrected_interactions_filename = root + data_dir + \
        "/corrected_interactions/" + pattern_name + "_corrected_interactions.csv"
        corrected_interactions = pd.read_csv(corrected_interactions_filename,
                                             sep=",", 
                                             encoding="utf-8")

        for iinter in range(corrected_interactions.shape[0]):
            protein_dbid = corrected_interactions["UniprotID"][iinter]
            drug_dbid = corrected_interactions["DrugbankID"][iinter]
            corrected_interaction_bool = corrected_interactions[ "corrected_interaction_bool"][iinter]
            
            corrected_DB = correct_interactions(protein_dbid,
                                                drug_dbid,
                                                corrected_interaction_bool,
                                                corrected_DB)

    for seed in list_seed:
        print("seed:", seed)

        # Create the train dataset
        train_dataset = get_train_dataset(seed, corrected_DB)
        true_inter = train_dataset.true_inter
        false_inter = train_dataset.false_inter

        # Compute the kernel of interactions
        K_train = make_K_train(train_dataset, corrected_DB, kernels)
        y_train = np.concatenate((true_inter.interaction_bool, 
                                  false_inter.interaction_bool),
                                  axis=0)

        print("Training dataset's kernel of interactions prepared.")

        # Create the classifier
        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_train, y_train.ravel())
        list_clf.append(clf)

        # the list of couples in the train set are necessary to compute the 
        # similarity kernel for the interactions that we want to predict 
        # true_inter = train_set[2]
        # false_inter = train_set[3]
        list_couples = true_inter.list_couples + false_inter.list_couples
        list_couples_of_clf.append(list_couples)
    
    print("Classifiers done.")
        
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