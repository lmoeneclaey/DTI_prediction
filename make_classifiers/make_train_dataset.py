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

def make_train_dataset(preprocessed_DB):
    """ 
    Get the list of all the couples that are in the train:
        - the "positive" (known) interactions (with indices ind_true_inter)
        _ the "negative" (unknown) interactions (with indices ind_false_inter)  

    Parameters
    ----------
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    all_clf_train_interactions_arr : List of numpy array of the train interactions

    """ 

    dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot
    intMat = preprocessed_DB.intMat
    interactions = preprocessed_DB.interactions

    # "POSITIVE" INTERACTIONS

    train_positive_interactions = copy.deepcopy(interactions)
    train_positive_interactions_pd = pd.DataFrame(train_positive_interactions.array, 
                                                  columns=['UniProt ID', 
                                                            'Drugbank ID', 
                                                            'interaction_bool'])

    proteins_count = dict(train_positive_interactions_pd['UniProt ID'].value_counts())

    # "NEGATIVE" INTERACTIONS

    # get the interactions indices
    # ind_false_inter : indices where there is not an interaction
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

    # number of classifiers
    nb_clf = 5

    all_clf_train_interactions_arr = []

    for iclf in range(nb_clf):

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

        train_interactions_one_clf_pd = pd.concat([train_positive_interactions_pd, train_negative_interactions_one_clf_pd])
        all_clf_train_interactions_arr.append(train_interactions_one_clf_pd.to_numpy())

    print("Train datasets preprared.")

    return all_clf_train_interactions_arr

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

    list_train_datasets_array = make_train_dataset(corrected_DB)

    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'

    pickle.dump(list_train_datasets_array, 
                open(train_datasets_array_filename, 'wb'), 
                protocol=2)
    
    print("Train datasets prepared.")

