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

# Balanced on proteins and drugs
def make_nested_folds(seed, DB):
    """ 
    Parameters
    ----------
    seed : number
    DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    """ 

    dict_ind2mol = DB.drugs.dict_ind2mol
    dict_ind2prot = DB.proteins.dict_ind2prot
    intMat = DB.intMat
    interactions = DB.interactions

    nb_folds = 6

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    shuffled = train_positive_interactions_pd.sample(frac=1)
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

    # loop on number of clfs if you want many classifiers

    # loop on number fo folds

    remaining_negative_interactions_pd = copy.deepcopy(all_negative_interactions_pd)

    interactions_arr_list = []

    for ifold in range(6):
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
            negative_interactions_one_prot_pd = remaining_negative_interactions_pd[remaining_negative_interactions_pd['UniProt ID']==protein_id]
            
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

            possible_frequent_hitters_negative_interactions_one_prot_pd = negative_interactions_one_prot_pd[negative_interactions_one_prot_pd['Drugbank ID'].isin(possible_frequent_hitters)]
            nb_frequent_hitters_negative_interactions = possible_frequent_hitters_negative_interactions_one_prot_pd.shape[0]
            print("possible frequent hitters negative interactions:", nb_frequent_hitters_negative_interactions)

            if nb_positive_interactions<nb_frequent_hitters_negative_interactions: 

                negative_interactions_one_prot_pd = possible_frequent_hitters_negative_interactions_one_prot_pd.sample(nb_positive_interactions)

            else:
                
                frequent_hitters_negative_interactions_one_prot_pd = possible_frequent_hitters_negative_interactions_one_prot_pd
                
                nb_negative_interactions_remaining = nb_positive_interactions - nb_frequent_hitters_negative_interactions
                
                possible_one_int_negative_interactions_one_prot_pd = negative_interactions_one_prot_pd[negative_interactions_one_prot_pd['Drugbank ID'].isin(possible_one_int_drug)]
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

        interactions_arr_list.append(np.array(interactions_one_fold_pd))

    return interactions_arr_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Create train datasets for a particular dataset and save it.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

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

    if not os.path.exists(root + data_dir + '/' + 'classifiers/train_datasets'):
        os.mkdir(root + data_dir + '/' + 'classifiers/train_datasets')
        print("Train dataset directory for ", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("Train dataset directory for ", pattern_name, ",", args.DB_version,
        "already exists.")

    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)

    print("Initially, there are", preprocessed_DB.interactions.nb, "interactions \
        in the preprocessed database.")

    corrected_DB = copy.deepcopy(preprocessed_DB)

    if args.orphan is not None:
        for dbid in args.orphan:
            corrected_DB = get_orphan(DB=corrected_DB, dbid=dbid)

    if args.correct_interactions == True:

        corrected_interactions_filename = root + data_dir + \
        "/corrected_interactions/" + pattern_name + "_corrected_interactions.csv"
        corrected_interactions = pd.read_csv(corrected_interactions_filename,
                                             sep=",", 
                                             encoding="utf-8")
        nb_interactions_to_correct = corrected_interactions.shape[0]
        print(nb_interactions_to_correct, " interactions to add or correct.")

        for iinter in range(nb_interactions_to_correct):
            protein_dbid = corrected_interactions["UniprotID"][iinter]
            drug_dbid = corrected_interactions["DrugbankID"][iinter]
            corrected_interaction_bool = corrected_interactions[ "corrected_interaction_bool"][iinter]
            
            corrected_DB = correct_interactions(protein_dbid,
                                                drug_dbid,
                                                corrected_interaction_bool,
                                                corrected_DB)

    print("For this classifier, there will be", corrected_DB.interactions.nb, 
          "interactions.")

    list_nested_train_datasets_array = make_nested_folds(seed = 242, DB = corrected_DB)

    nested_train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_nested_train_datasets_array.data'

    pickle.dump(list_nested_train_datasets_array, 
                open(nested_train_datasets_array_filename, 'wb'), 
                protocol=2)
    
    print("Nested train datasets prepared.")