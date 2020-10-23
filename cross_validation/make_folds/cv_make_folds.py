import argparse
import numpy as np
import os
import pandas as pd
import pickle

import sklearn.model_selection as model_selection

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB, get_subset_couples
from DTI_prediction.process_dataset.process_DB import get_DB

root = './../CFTR_PROJECT/'

# version du 23/06/2020
# def make_folds(seed, nb_clf, preprocessed_DB):
#     """ 
#     Parameters
#     ----------
#     seed : number
#     preprocessed_DB : tuple of length 8
#         got with the function process_dataset.process_DB.get_DB()

#     Returns
#     -------
#     list_couples : list
#         List of all the couples in the train data set
#     y : list
#         results to predict
#     ind_true_inter : np.array
#     ind_false_inter : np.array
#     """ 

#     dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
#     dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot
#     intMat = preprocessed_DB.intMat
#     interactions = preprocessed_DB.interactions

#     # Set the different seeds
#     nb_folds = 5
#     true_seed = 62
#     false_seed = 53

#     # "TRUE" INTERACTIONS (TRAIN AND TEST)
#     nb_true_inter = preprocessed_DB.interactions.nb

#     # initialisation
#     # train_true_folds is a list (length: nb_folds) of Couples (length: ((nb_folds-1)/nb_folds)*nb_true_inter)
#     train_true_folds = []
#     # test_true_folds is a list (length: nb_folds) of Couples (length: (1/nb_folds)*nb_true_inter)
#     test_true_folds = []

#     skf_true = model_selection.KFold(shuffle=True,random_state=true_seed)
#     for train_index, test_index in skf_true.split(range(nb_true_inter)):
    
#         train_fold = get_subset_couples(interactions,
#                                         train_index)

#         test_fold = get_subset_couples(interactions,
#                                        test_index)
    
#         train_true_folds.append(train_fold)
#         test_true_folds.append(test_fold)

#     # "FALSE" INTERACTIONS
        
#     # get the interactions indices
#     # ind_false_inter : indices where there is not an interaction
#     ind_all_false_inter = np.where(intMat == 0)
#     nb_all_false_inter = len(ind_all_false_inter[0])

#     # choose between all the "false" interactions, nb_true_interactions couples,
#     # without replacement
#     np.random.seed(seed)
#     mask = np.random.choice(np.arange(nb_all_false_inter), 
#                             (nb_clf+1)*nb_true_inter,
#                             replace=False)
#     mask_test = mask[:nb_true_inter]
#     mask_train = mask[nb_true_inter:]

#     ## TEST "FALSE" INTERACTIONS

#     # get a new list with only the "false" interactions indices which will be \
#     # in the train data set
#     ind_test_false_inter = (ind_all_false_inter[0][mask_test], 
#                             ind_all_false_inter[1][mask_test])
#     nb_test_false_inter = len(ind_test_false_inter[0])

#     list_test_false_inter = []
#     for i in range(nb_test_false_inter):
#         list_test_false_inter.append((dict_ind2prot[ind_test_false_inter[0][i]],
#                                     dict_ind2mol[ind_test_false_inter[1][i]]))
#     list_test_false_inter_arr = np.array(list_test_false_inter)

#     # initialisation
#     # test_false_folds is a list (length: nb_folds) of Couples (length: (1/nb_folds)*nb_true_inter)
#     test_false_folds = []

#     skf_false = model_selection.KFold(nb_folds, shuffle=True, random_state=false_seed)
#     for train_index, test_index in skf_false.split(range(nb_test_false_inter)):
    
#         test_fold = Couples(list_couples=list_test_false_inter_arr[test_index].tolist(),
#                             interaction_bool=np.array([0]*len(test_index)).reshape(-1,1))
    
#         test_false_folds.append(test_fold)

#     ## TRAIN "FALSE" INTERACTIONS

#     # initialisation
#     # train_false_folds is a list (length: nb_clfs) of Couples (length: ((nb_folds-1)/nb_folds)*nb_true_inter)
#     train_false_folds = []
    
#     # get the number of true interaction in one train fold
#     nb_true_inter_per_clf = np.int(((nb_folds-1)/nb_folds)*nb_true_inter)

#     # get (nb_clf) samples of mask_train of length nb_true_inter_per_clf
#     mask_train_per_clf = []
#     for iclf in range(nb_clf):
#         mask_train_per_clf.append(mask_train[(iclf*nb_true_inter_per_clf):((iclf+1)*nb_true_inter_per_clf)])
    
#     for iclf in range(nb_clf):
    
#         mask_train = mask_train_per_clf[iclf]
    
#         ind_train_false_inter = (ind_all_false_inter[0][mask_train], 
#                                  ind_all_false_inter[1][mask_train])
#         nb_train_false_inter = len(ind_train_false_inter[0])

#         list_train_false_inter = []
#         for i in range(nb_train_false_inter):
#             list_train_false_inter.append((dict_ind2prot[ind_train_false_inter[0][i]],
#                                         dict_ind2mol[ind_train_false_inter[1][i]]))
    
#         train_fold = Couples(list_couples=list_train_false_inter,
#                              interaction_bool=np.array([0]*nb_train_false_inter).reshape(-1,1))
        
#         train_false_folds.append(train_fold)

#     print("Train folds and test folds prepared.")

#     return train_true_folds, test_true_folds, train_false_folds, test_false_folds

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
    nb_folds = 6
    positive_seed = 62
    negative_seed = 53

    # "POSITIVE" INTERACTIONS (TRAIN AND TEST)
    nb_positive_inter = preprocessed_DB.interactions.nb

    # initialisation
    # train_positive_folds is a list (length: nb_folds) of Couples (length: ((nb_folds-1)/nb_folds)*nb_positive_inter)
    train_positive_folds_pd = []
    # test_positive_folds is a list (length: nb_folds) of Couples (length: (1/nb_folds)*nb_positive_inter)
    test_positive_folds_pd = []

    skf_positive = model_selection.KFold(shuffle=True,random_state=positive_seed, n_splits=nb_folds)
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
    
    for ifold in range(nb_folds):

        # get for each protein in the train dataset the number of occurences
        train_positive_fold_pd = train_positive_folds_pd[ifold]
        proteins_count = dict(train_positive_fold_pd['UniProt ID'].value_counts())
        drugs_count = dict(train_positive_fold_pd['Drugbank ID'].value_counts())


        # all of the classifiers train negative interactions for one fold
        # useful to have the test negative interactions
        all_clf_train_negative_interactions_one_fold_pd = []
        # all of the classifiers train interactions for one fold
        all_clf_train_interactions_one_fold_arr = []

        for iclf in range(nb_clf):
            
            # list of all the train negative interactions for one classifier
            all_prot_train_negative_interactions_one_clf_pd = []

            for row_nb in range(len(proteins_count)):

            

                protein_id = list(proteins_count.keys())[row_nb]
                nb_positive_interactions_in_train = proteins_count[protein_id]

                print(nb_positive_interactions_in_train)

                # get all the negative interactions concerning this protein
                negative_interactions_one_prot_pd = all_negative_interactions_pd[all_negative_interactions_pd['UniProt ID']==protein_id]

                # remove the interactions involving drugs "already full in number of negative interactions"
                possible_drugs = []
                for (drug, count) in drugs_count.items():
                    if count!=0:
                        possible_drugs.append(drug)

                possible_negative_interactions_one_prot_pd = negative_interactions_one_prot_pd[negative_interactions_one_prot_pd['Drugbank ID'].isin(possible_drugs)]

                print(possible_negative_interactions_one_prot_pd.shape[0])

                # sample among the negative interactions concerning this prot,
                # the number of positive interactions in the train dataset
                train_negative_interactions_one_prot_one_clf_pd = possible_negative_interactions_one_prot_pd.sample(nb_positive_interactions_in_train)
                all_prot_train_negative_interactions_one_clf_pd.append(train_negative_interactions_one_prot_one_clf_pd)

                # Change the number of counts in the drug dictionary
                for drug_id in train_negative_interactions_one_prot_one_clf_pd['Drugbank ID']:
                    dict_old_count = drugs_count[drug_id]
                    drugs_count[drug_id] = dict_old_count - 1
        
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

    # train_true_folds = folds[0]
    # test_true_folds = folds[1]
    # train_false_folds = folds[2]
    # test_false_folds = folds[3]
    train_folds_arr = folds[0]
    test_folds_arr = folds[1]

    # nb_folds = len(train_true_folds)

    # Save test folds

    test_folds_array_filename = cv_dirname + 'test_folds/' \
        + args.DB_type + '_test_folds_array.data'

    # test_folds = []
    # test_folds_array = []
    # for ifold in range(nb_folds):

    #     test_folds.append(test_true_folds[ifold] + test_false_folds[ifold])
    #     test_folds_array.append(test_folds[ifold].array)

    # pickle.dump(test_folds_array, 
    #             open(test_folds_array_filename, 'wb'))

    pickle.dump(test_folds_arr, 
                open(test_folds_array_filename, 'wb'))

    # Save train folds

    # train_true_folds_array_filename = cv_dirname + 'train_folds/' + \
    #     args.DB_type + '_train_true_folds_' + str(args.nb_clf) + '_clf_array.data'
    # train_false_folds_array_filename = cv_dirname + 'train_folds/' + \
    #     args.DB_type + '_train_false_folds_' + str(args.nb_clf) + '_clf_array.data'

    # train_true_folds_array = []
    # for ifold in range(nb_folds):
    #     train_true_folds_array.append(train_true_folds[ifold].array)

    # train_false_folds_array = []
    # for iclf in range(args.nb_clf):
    #     train_false_folds_array.append(train_false_folds[iclf].array)

    # pickle.dump(train_true_folds_array, 
    #             open(train_true_folds_array_filename, 'wb'))

    # pickle.dump(train_false_folds_array, 
    #             open(train_false_folds_array_filename, 'wb'))

    train_folds_array_filename = cv_dirname + 'train_folds/' + \
        args.DB_type + '_train_folds_array.data'

    pickle.dump(train_folds_arr, 
                open(train_folds_array_filename, 'wb'))