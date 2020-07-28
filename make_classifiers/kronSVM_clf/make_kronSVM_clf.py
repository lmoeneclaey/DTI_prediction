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
# from DTI_prediction.process_dataset.correct_interactions import get_orphan, correct_interactions
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

# from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset, get_train_dataset, make_K_train
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

    # parser.add_argument("--orphan", type = str, action='append',
    #                     help = "molecules which you want to orphanize in the \
    #                         train data set")

    # parser.add_argument("--correct_interactions", default = False, action="store_true",
    #                     help = "whether or not to add or correct some \
    #                         interactions, False by default")

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

    # print("Initially, there are", preprocessed_DB.interactions.nb, "interactions \
    #     in the preprocessed database.")

    # corrected_DB = copy.deepcopy(preprocessed_DB)

    # for dbid in args.orphan:
    #     corrected_DB = get_orphan(DB=corrected_DB, dbid=dbid)

    # if args.correct_interactions == True:

    #     corrected_interactions_filename = root + data_dir + \
    #     "/corrected_interactions/" + pattern_name + "_corrected_interactions.csv"
    #     corrected_interactions = pd.read_csv(corrected_interactions_filename,
    #                                          sep=",", 
    #                                          encoding="utf-8")
    #     nb_interactions_to_correct = corrected_interactions.shape[0]
    #     print(nb_interactions_to_correct, " interactions to add or correct.")

    #     for iinter in range(nb_interactions_to_correct):
    #         protein_dbid = corrected_interactions["UniprotID"][iinter]
    #         drug_dbid = corrected_interactions["DrugbankID"][iinter]
    #         corrected_interaction_bool = corrected_interactions[ "corrected_interaction_bool"][iinter]
            
    #         corrected_DB = correct_interactions(protein_dbid,
    #                                             drug_dbid,
    #                                             corrected_interaction_bool,
    #                                             corrected_DB)

    # print("For this classifier, there will be", corrected_DB.interactions.nb, 
    #       "interactions.")

    # list_seed = [2821, 1148, 1588, 188, 933]
    # list_seed = [1177, 2126, 1841,  361, 2462]
    # list_seed = [3082, 1506, 3426, 2446,  609] # 07-07-2020
    # list_seed = [1020, 2521, 3362, 357, 935] # 08-07-2020
    # list_couples_of_clf = []
    # list_train_datasets = []

    # for seed in list_seed:
        # print("seed:", seed)

    list_clf = []

    for iclf in range(nb_clf):

        # Create the train dataset
        # train_dataset = get_train_dataset(seed, corrected_DB)
        # true_inter = train_dataset.true_inter
        # false_inter = train_dataset.false_inter
        train_dataset = list_train_datasets[iclf]

        # Compute the kernel of interactions
        K_train = make_K_train(train_dataset, preprocessed_DB, kernels)
        # y_train = np.concatenate((true_inter.interaction_bool, 
        #                           false_inter.interaction_bool),
        #                           axis=0)
        y_train = train_dataset.interaction_bool

        print("Training dataset's kernel of interactions prepared.")

        # Create the classifier
        clf = SVC(C=C, 
                  kernel='precomputed', 
                  probability=True, 
                  class_weight='balanced')
        clf.fit(K_train, y_train.ravel())
        list_clf.append(clf)

        # the list of couples in the train set are necessary to compute the 
        # similarity kernel for the interactions that we want to predict 
        # true_inter = train_set[2]
        # false_inter = train_set[3]
        # list_couples = true_inter.list_couples + false_inter.list_couples
        # list_couples_of_clf.append(list_couples)
        # train_dataset = np.concatenate((true_inter.array, 
        #                                false_inter.array), 
        #                                axis=0)
        # list_train_datasets.append(train_dataset)
    
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

    # Couples of the classifier
    # couples_filename = clf_dirname + pattern_name + \
    # '_kronSVM_list_couples_of_clf.data'

    # pickle.dump(list_couples_of_clf, 
    #             open(couples_filename, 'wb'), 
    #             protocol=2)

    # Train datasets of the classifier
    # train_datasets_filename = clf_dirname + pattern_name + \
    #     '_kronSVM_list_train_datasets.data'

    # pickle.dump(list_train_datasets, 
    #             open(train_datasets_filename, 'wb'), 
    #             protocol=2)
    
    print("Classifiers saved.")