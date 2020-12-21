import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB, get_subset_couples
from DTI_prediction.process_dataset.process_DB import get_DB

from DTI_prediction.process_dataset.correct_interactions import get_orphan, correct_interactions

root = './../CFTR_PROJECT/'

# non balanced
def make_nested_folds_non_balanced(DB):

    """ 
    Parameters
    ----------
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    """ 

    dict_ind2mol = DB.drugs.dict_ind2mol
    dict_ind2prot = DB.proteins.dict_ind2prot
    intMat = DB.intMat
    interactions = DB.interactions

    nb_folds = 5

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    shuffled = train_positive_interactions_pd.sample(frac=1, 
                                                     random_state=54)
    result = np.array_split(shuffled, nb_folds)

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
                                                'interaction_bool':'0'})

    all_negative_interactions_pd.index.name = 'neg_int_id'
    all_negative_interactions_pd.reset_index(inplace=True)

    nb_clf = 5

    interactions_arr_list_all_clf = []
    for iclf in range(nb_clf):

        remaining_negative_interactions_pd = copy.deepcopy(all_negative_interactions_pd)
        interactions_arr_list_per_clf = []

        for ifold in range(nb_folds):

            # get for each drug in the train dataset the number of occurences
            positive_fold_pd = result[ifold]

            nb_positive_interactions = positive_fold_pd.shape[0]
            all_drug_negative_interactions_pd = remaining_negative_interactions_pd.sample(nb_positive_interactions)
            remaining_negative_interactions_pd = remaining_negative_interactions_pd[~remaining_negative_interactions_pd.neg_int_id.isin(all_drug_negative_interactions_pd.neg_int_id)]

            interactions_one_fold_pd = pd.concat([positive_fold_pd, all_drug_negative_interactions_pd[['UniProt ID', 'Drugbank ID', 'interaction_bool']]])
            print("Fold n", ifold, "done.")

            interactions_arr_list_per_clf.append(np.array(interactions_one_fold_pd))
        
        interactions_arr_list_all_clf.append(interactions_arr_list_per_clf)
        
    return interactions_arr_list_all_clf

# balanced on drugs
def make_nested_folds_balanced_on_drugs(DB):

    """ 
    Parameters
    ----------
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    """ 

    dict_ind2mol = DB.drugs.dict_ind2mol
    dict_ind2prot = DB.proteins.dict_ind2prot
    intMat = DB.intMat
    interactions = DB.interactions

    nb_folds = 5

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    shuffled = train_positive_interactions_pd.sample(frac=1, 
                                                     random_state=54)
    result = np.array_split(shuffled, nb_folds)

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
                                                'interaction_bool':'0'})

    all_negative_interactions_pd.index.name = 'neg_int_id'
    all_negative_interactions_pd.reset_index(inplace=True)

    nb_clf = 5

    interactions_arr_list_all_clf = []
    for iclf in range(nb_clf):

        remaining_negative_interactions_pd = copy.deepcopy(all_negative_interactions_pd)
        interactions_arr_list_per_clf = []

        for ifold in range(nb_folds):

            # all of the folds negative interactions for one classifier
            # useful to have the remaining negative interactions
            all_drug_negative_interactions_list = []

            # get for each drug in the train dataset the number of occurences
            positive_fold_pd = result[ifold]
            drugs_count = dict(positive_fold_pd['Drugbank ID'].value_counts())

            for row_nb in range(len(drugs_count)):

                print("row_nb:", row_nb)

                drug_id = list(drugs_count.keys())[row_nb]
                nb_positive_interactions = drugs_count[drug_id]

                # get all the negative interactions concerning this drug
                possible_negative_interactions_one_drug_pd = remaining_negative_interactions_pd[remaining_negative_interactions_pd['Drugbank ID']==drug_id]

                # sample among the negative interactions concerning this drug,
                # the number of positive interactions in the train dataset
                negative_interactions_one_drug_pd = possible_negative_interactions_one_drug_pd.sample(nb_positive_interactions)
                    
                all_drug_negative_interactions_list.append(negative_interactions_one_drug_pd)

            all_drug_negative_interactions_pd = pd.concat(all_drug_negative_interactions_list)
            remaining_negative_interactions_pd = remaining_negative_interactions_pd[~remaining_negative_interactions_pd.neg_int_id.isin(all_drug_negative_interactions_pd.neg_int_id)]

            interactions_one_fold_pd = pd.concat([positive_fold_pd, all_drug_negative_interactions_pd[['UniProt ID', 'Drugbank ID', 'interaction_bool']]])
            print("Fold n", ifold, "done.")

            interactions_arr_list_per_clf.append(np.array(interactions_one_fold_pd))

        interactions_arr_list_all_clf.append(interactions_arr_list_per_clf)

    return interactions_arr_list_all_clf

# balanced on proteins 
def make_nested_folds_balanced_on_proteins(DB):

    """ 
    Parameters
    ----------
    seed : number
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    """ 

    dict_ind2mol = DB.drugs.dict_ind2mol
    dict_ind2prot = DB.proteins.dict_ind2prot
    intMat = DB.intMat
    interactions = DB.interactions

    nb_folds = 5

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    shuffled = train_positive_interactions_pd.sample(frac=1, 
                                                     random_state=54)
    result = np.array_split(shuffled, nb_folds)

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
                                                'interaction_bool':'0'})

    all_negative_interactions_pd.index.name = 'neg_int_id'
    all_negative_interactions_pd.reset_index(inplace=True)

    nb_clf = 5

    interactions_arr_list_all_clf = []
    for iclf in range(nb_clf):

        remaining_negative_interactions_pd = copy.deepcopy(all_negative_interactions_pd)
        interactions_arr_list_per_clf = []

        for ifold in range(nb_folds):

            # all of the folds negative interactions for one classifier
            # useful to have the remaining negative interactions
            all_prot_negative_interactions_list = []

            # get for each protein in the train dataset the number of occurences
            positive_fold_pd = result[ifold]
            proteins_count = dict(positive_fold_pd['UniProt ID'].value_counts())

            for row_nb in range(len(proteins_count)):

                print("row_nb:", row_nb)

                protein_id = list(proteins_count.keys())[row_nb]
                nb_positive_interactions = proteins_count[protein_id]

                # get all the negative interactions concerning this protein
                possible_negative_interactions_one_prot_pd = remaining_negative_interactions_pd[remaining_negative_interactions_pd['UniProt ID']==protein_id]

                # sample among the negative interactions concerning this prot,
                # the number of positive interactions in the train dataset
                negative_interactions_one_prot_pd = possible_negative_interactions_one_prot_pd.sample(nb_positive_interactions)
                    
                all_prot_negative_interactions_list.append(negative_interactions_one_prot_pd)

            all_prot_negative_interactions_pd = pd.concat(all_prot_negative_interactions_list)
            remaining_negative_interactions_pd = remaining_negative_interactions_pd[~remaining_negative_interactions_pd.neg_int_id.isin(all_prot_negative_interactions_pd.neg_int_id)]

            interactions_one_fold_pd = pd.concat([positive_fold_pd, all_prot_negative_interactions_pd[['UniProt ID', 'Drugbank ID', 'interaction_bool']]])
            print("Fold n", ifold, "done.")

            interactions_arr_list_per_clf.append(np.array(interactions_one_fold_pd))

        interactions_arr_list_all_clf.append(interactions_arr_list_per_clf)

    return interactions_arr_list_all_clf

# version du 12/10/2020
# Balanced on proteins and drugs
def make_nested_folds_double_balanced(DB):

    """ 
    Parameters
    ----------
    seed : number
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    """ 

    dict_ind2mol = DB.drugs.dict_ind2mol
    dict_ind2prot = DB.proteins.dict_ind2prot
    intMat = DB.intMat
    interactions = DB.interactions

    nb_folds = 5

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    shuffled = train_positive_interactions_pd.sample(frac=1, 
                                                     random_state=54)
    result = np.array_split(shuffled, nb_folds)

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
                                                'interaction_bool':'0'})

    all_negative_interactions_pd.index.name = 'neg_int_id'
    all_negative_interactions_pd.reset_index(inplace=True)

    nb_clf = 5

    interactions_arr_list_all_clf = []
    for iclf in range(nb_clf):

        remaining_negative_interactions_pd = copy.deepcopy(all_negative_interactions_pd)
        interactions_arr_list_per_clf = []

        for ifold in range(nb_folds):
            #ifold = 0

            # all of the folds negative interactions for one classifier
            # useful to have the remaining negative interactions
            all_prot_negative_interactions_list = []

            # get for each protein in the train dataset the number of occurences
            positive_fold_pd = result[ifold]
            proteins_count = dict(positive_fold_pd['UniProt ID'].value_counts())
            drugs_count = dict(positive_fold_pd['Drugbank ID'].value_counts())
            remaining_drugs_count = copy.deepcopy(drugs_count)

            for row_nb in range(len(proteins_count)):

                print("row_nb:", row_nb)

                protein_id = list(proteins_count.keys())[row_nb]
                nb_positive_interactions = proteins_count[protein_id]

                # get all the negative interactions concerning this protein
                possible_negative_interactions_one_prot_pd = remaining_negative_interactions_pd[remaining_negative_interactions_pd['UniProt ID']==protein_id]
                
                # remove the interactions involving drugs "already full in number of negative interactions"
                possible_drugs = []
                possible_frequent_hitters = []
                possible_one_int_drug = []
                for (drug, count) in remaining_drugs_count.items():
                    if count!=0:
                        possible_drugs.append(drug)
                        if count>1:
                            possible_frequent_hitters.append(drug)
                        else:
                            possible_one_int_drug.append(drug)
                
                print("number of drugs remaining:", len(possible_drugs), ":", len(possible_frequent_hitters), "frequent hitters and", len(possible_one_int_drug), "only one.")

                # sample among the negative interactions concerning this prot,
                # the number of positive interactions in the train dataset

                possible_frequent_hitters_negative_interactions_one_prot_pd = possible_negative_interactions_one_prot_pd[possible_negative_interactions_one_prot_pd['Drugbank ID'].isin(possible_frequent_hitters)]
                nb_frequent_hitters_negative_interactions = possible_frequent_hitters_negative_interactions_one_prot_pd.shape[0]
                print("possible frequent hitters negative interactions:", nb_frequent_hitters_negative_interactions)

                if nb_positive_interactions<nb_frequent_hitters_negative_interactions: 

                    negative_interactions_one_prot_pd = possible_frequent_hitters_negative_interactions_one_prot_pd.sample(nb_positive_interactions)

                else:
                    
                    frequent_hitters_negative_interactions_one_prot_pd = possible_frequent_hitters_negative_interactions_one_prot_pd
                    
                    nb_negative_interactions_remaining = nb_positive_interactions - nb_frequent_hitters_negative_interactions
                    
                    possible_one_int_negative_interactions_one_prot_pd = possible_negative_interactions_one_prot_pd[possible_negative_interactions_one_prot_pd['Drugbank ID'].isin(possible_one_int_drug)]
                    one_int_negative_interactions_one_prot_pd = possible_one_int_negative_interactions_one_prot_pd.sample(nb_negative_interactions_remaining)
                    
                    negative_interactions_one_prot_pd = pd.concat([frequent_hitters_negative_interactions_one_prot_pd,
                                                                one_int_negative_interactions_one_prot_pd])
                    
                all_prot_negative_interactions_list.append(negative_interactions_one_prot_pd)

                # Change the number of counts in the drug dictionary
                for drug_id in negative_interactions_one_prot_pd['Drugbank ID']:
                    dict_old_count = remaining_drugs_count[drug_id]
                    remaining_drugs_count[drug_id] = dict_old_count - 1

            all_prot_negative_interactions_pd = pd.concat(all_prot_negative_interactions_list)
            remaining_negative_interactions_pd = remaining_negative_interactions_pd[~remaining_negative_interactions_pd.neg_int_id.isin(all_prot_negative_interactions_pd.neg_int_id)]

            interactions_one_fold_pd = pd.concat([positive_fold_pd, all_prot_negative_interactions_pd[['UniProt ID', 'Drugbank ID', 'interaction_bool']]])
            print("Fold n", ifold, "done.")

            interactions_arr_list_per_clf.append(np.array(interactions_one_fold_pd))
        
        interactions_arr_list_all_clf.append(interactions_arr_list_per_clf)

    return interactions_arr_list_all_clf


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Create nested folds for a particular dataset and save it.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("--balanced_on_proteins", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--balanced_on_drugs", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    #create directories
    if not os.path.exists(root + data_dir + '/' + 'cross_validation'):
        os.mkdir(root + data_dir + '/' + 'cross_validation')
        os.mkdir(root + data_dir + '/' + 'cross_validation' + '/' + 'nested_folds')
        print("Cross validation directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("Cross validation directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'cross_validation/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    corrected_DB = copy.deepcopy(preprocessed_DB)

    if args.balanced_on_proteins == True:
        if args.balanced_on_drugs == True:
            folds_arr = make_nested_folds_double_balanced(preprocessed_DB)
            nested_folds_array_filename = cv_dirname + 'nested_folds/' \
        + args.DB_type + '_nested_folds_double_balanced_5_clf_array.data'
        else:
            folds_arr = make_nested_folds_balanced_on_proteins(preprocessed_DB)
            nested_folds_array_filename = cv_dirname + 'nested_folds/' \
        + args.DB_type + '_nested_folds_balanced_on_proteins_5_clf_array.data'
    else:
        if args.balanced_on_drugs == True:
            folds_arr = make_nested_folds_balanced_on_drugs(preprocessed_DB)
            nested_folds_array_filename = cv_dirname + 'nested_folds/' \
        + args.DB_type + '_nested_folds_balanced_on_drugs_5_clf_array.data'
        else:
            folds_arr = make_nested_folds_non_balanced(preprocessed_DB)
            nested_folds_array_filename = cv_dirname + 'nested_folds/' \
        + args.DB_type + '_nested_folds_non_balanced_5_clf_array.data'        

    pickle.dump(folds_arr, 
                open(nested_folds_array_filename, 'wb'))