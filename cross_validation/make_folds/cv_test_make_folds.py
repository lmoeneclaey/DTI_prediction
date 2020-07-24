import argparse
import numpy as np
import os
import pandas as pd
import pickle

import sklearn.model_selection as model_selection

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB, get_subset_couples
from DTI_prediction.process_dataset.process_DB import get_DB

root = './../CFTR_PROJECT/'

# version du 22/07/2020
def make_folds(seed, nb_clf, preprocessed_DB):
    """ 
    Parameters
    ----------
    seed : number
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    list_couples : list
        List of all the couples in the train data set
    y : list
        results to predict
    ind_true_inter : np.array
    ind_false_inter : np.array
    """ 

    dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot
    intMat = preprocessed_DB.intMat
    interactions = preprocessed_DB.interactions

    # Set the different seeds
    nb_folds = 5
    positive_seed = 62
    negative_seed = 53

    # "POSITIVE" INTERACTIONS (TRAIN AND TEST)
    nb_positive_inter = preprocessed_DB.interactions.nb

    # initialisation
    # train_positive_folds is a list (length: nb_folds) of Couples (length: ((nb_folds-1)/nb_folds)*nb_positive_inter)
    train_positive_folds_pd = []
    # test_positive_folds is a list (length: nb_folds) of Couples (length: (1/nb_folds)*nb_positive_inter)
    test_positive_folds_pd = []

    skf_positive = model_selection.KFold(shuffle=True,random_state=positive_seed)
    for train_index, test_index in skf_positive.split(range(nb_positive_inter)):
    
        train_positive_fold = get_subset_couples(interactions,
                                                 train_index)
        train_positive_fold_pd = pd.DataFrame(train_positive_fold.array, 
                                              columns=['UniProt ID', 
                                                       'Drugbank ID', 
                                                       'interaction_bool'])

        test_positive_fold = get_subset_couples(interactions,
                                    test_index)
        test_positive_fold_pd = pd.DataFrame(test_positive_fold.array, 
                                             columns=['UniProt ID', 
                                                      'Drugbank ID', 
                                                      'interaction_bool'])
    
        train_positive_folds_pd.append(train_positive_fold_pd)
        test_positive_folds_pd.append(test_positive_fold_pd)

    # "NEGATIVE" INTERACTIONS
        
    # get the negative interactions indices
    # ind_all_negative_inter : indices where there is not an interaction
    ind_all_negative_inter = np.where(intMat == 0)
    nb_all_negative_inter = len(ind_all_negative_inter[0])

    all_negative_interactions_protein_id = []
    all_negative_interactions_drug_id = []
    for row in range(nb_all_negative_inter):
        all_negative_interactions_protein_id.append(dict_ind2prot[ind_all_negative_inter[0][row]])
        all_negative_interactions_drug_id.append(dict_ind2mol[ind_all_negative_inter[1][row]])

    all_negative_interactions_pd = pd.DataFrame({'UniProt ID':all_negative_interactions_protein_id, 
                                                 'Drugbank ID':all_negative_interactions_drug_id,
                                                 'interaction_bool':0})

    train_folds_arr = []
    test_folds_arr = []
    
    for ifold in range(1):

        # get for each protein in the train dataset the number of occurences
        train_positive_fold_pd = train_positive_folds_pd[ifold]
        proteins_count = dict(train_positive_fold_pd['UniProt ID'].value_counts())

        # all of the classifiers train negative interactions for one fold
        # useful to have the test negative interactions
        all_clf_train_negative_interactions_one_fold_pd = []
        # all of the classifiers train interactions for one fold
        all_clf_train_interactions_one_fold_arr = []

        for iclf in range(1):
            
            # list of all the train negative interactions for one classifier
            all_prot_train_negative_interactions_one_clf_pd = []

            for row_nb in range(len(proteins_count)):
                protein_id = list(proteins_count.keys())[row_nb]
                nb_positive_interactions_in_train = proteins_count[protein_id]

                # get all the negative interactions concerning this protein
                negative_interactions_one_prot_pd = all_negative_interactions_pd[all_negative_interactions_pd['UniProt ID']==protein_id]

                # sample among the negative interactions concerning this prot,
                # the number of positive interactions in the train dataset
                train_negative_interactions_one_prot_one_clf_pd = negative_interactions_one_prot_pd.sample(nb_positive_interactions_in_train)
                all_prot_train_negative_interactions_one_clf_pd.append(train_negative_interactions_one_prot_one_clf_pd)
        
            train_negative_interactions_one_clf_pd = pd.concat(all_prot_train_negative_interactions_one_clf_pd)
            # all of the classifiers train negative interactions for one fold
            # useful to have the test negative interactions
            all_clf_train_negative_interactions_one_fold_pd.append(train_negative_interactions_one_clf_pd)
        
            train_interactions_one_clf_pd = pd.concat([train_positive_fold_pd, train_negative_interactions_one_clf_pd])
            # all of the classifiers train interactions for one fold
            all_clf_train_interactions_one_fold_arr.append(train_interactions_one_clf_pd.to_numpy())

            print("Classifier n", iclf, "done.")

        train_folds_arr.append(all_clf_train_interactions_one_fold_arr)
    
        # Get the negative interactions for the test fold 
        all_fold_train_negative_interactions_pd = pd.concat(all_clf_train_negative_interactions_one_fold_pd)
        unique_all_fold_train_negative_interactions_pd = all_fold_train_negative_interactions_pd.drop_duplicates()
        remain_train_negative_interactions_for_test_fold_pd = all_negative_interactions_pd.drop(unique_all_fold_train_negative_interactions_pd.index, axis=0)
        
        test_negative_one_fold_pd = remain_train_negative_interactions_for_test_fold_pd.sample(test_positive_folds_pd[ifold].shape[0])
        # Aggregate with the positive interactions to finalise the test fold
        test_fold_pd = pd.concat([test_positive_folds_pd[ifold], test_negative_one_fold_pd])
        test_folds_arr.append(test_fold_pd.to_numpy())

        print("Fold n", ifold, "done.")

    print("Train folds and test folds prepared.")

    return train_folds_arr, test_folds_arr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Create train and test folds for a particular dataset and save it.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("nb_clf", type = int,
                        help = "number of classifiers for future predictions, example = 5")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    #create directories
    if not os.path.exists(root + data_dir + '/' + 'cross_validation'):
        os.mkdir(root + data_dir + '/' + 'cross_validation')
        os.mkdir(root + data_dir + '/' + 'cross_validation' + '/' + 'test_folds')
        os.mkdir(root + data_dir + '/' + 'cross_validation' + '/' + 'train_folds')
        print("Cross validation directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("Cross validation directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'cross_validation/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    seed = 242
    folds = make_folds(seed, args.nb_clf, preprocessed_DB)

    train_folds_arr = folds[0]
    test_folds_arr = folds[1]

    # Save test folds

    test_folds_array_filename = cv_dirname + 'test_folds/' \
        + args.DB_type + '_test_folds_array_test_test.data'

    pickle.dump(test_folds_arr, 
                open(test_folds_array_filename, 'wb'))

    train_folds_array_filename = cv_dirname + 'train_folds/' + \
        args.DB_type + '_train_folds_array_test_test.data'

    pickle.dump(train_folds_arr, 
                open(train_folds_array_filename, 'wb'))