import argparse 
import csv
import collections
import copy
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys

root = '../CFTR_PROJECT/'

from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.utils.train_dataset_utils import get_number_of_interactions_per_mol, get_number_of_interactions_per_prot


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions with a kronSVM classifier."

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    args = parser.parse_args()

    DB = get_DB(args.DB_version, args.DB_type)

    # pattern_name variable
    pattern_name = args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name + '/'

    pred_dirname = "../" + pattern_name + '/predictions/kronSVM/'
    pred_stats_dirname = root + data_dir + '/predictions/kronSVM/'
    
    non_frequent_hitters_targets_pd = pd.read_csv("non_frequent_hitters_targets.txt", header=None)
    non_frequent_hitters_targets = list(non_frequent_hitters_targets_pd[0])

    all_predictions_files = []
    for i in range(DB.drugs.nb):

        dbid = DB.drugs.dict_ind2mol[i]

        predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + dbid + '_pred_clean_20201031.csv'
        if not(os.path.exists(predictions_filename)):
            predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + dbid + '_pred_clean_20201101.csv'
            if not(os.path.exists(predictions_filename)):
                predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + dbid + '_pred_clean_20201102.csv'

        if os.path.exists(predictions_filename):
            predictions = pd.read_csv(predictions_filename, encoding='utf-8')
            predictions['DrugBank ID']=dbid
            
            all_predictions_files.append(predictions)
            continue

    all_predictions = pd.concat(all_predictions_files)

    raw_data_dir = 'data/' + args.DB_version + '/raw/'
    raw_protein_df = pd.read_csv(root + raw_data_dir + \
                                 'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                                 sep=",")
    raw_protein_df = raw_protein_df.fillna('')

    interactions_pd = pd.DataFrame(DB.interactions.array, columns=['UniProt ID', 'DrugBank ID', 'interaction_bool'])

    # interactions_pd

    nb_interactions_per_prot_list = []
    for prot_id in list(DB.proteins.dict_ind2prot.values()):
        prot_interactions = interactions_pd[interactions_pd['UniProt ID']==prot_id]
        nb_interactions_per_prot_list.append(prot_interactions.shape[0])

    nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(DB.proteins.dict_ind2prot.values()),
                                            'Nb_interactions_per_prot':nb_interactions_per_prot_list})

    nb_interactions_per_prot = nb_interactions_per_prot.sort_values(by=['Nb_interactions_per_prot'], ascending=False)

    final_nb_interactions_per_prot = pd.merge(nb_interactions_per_prot,
                                    raw_protein_df[['UniProt ID', 'Gene Name', 'Name']],
                                    left_on='UniProt ID',
                                    right_on='UniProt ID')

    final_nb_interactions_per_prot = final_nb_interactions_per_prot.drop_duplicates()

    category = []
    for val in final_nb_interactions_per_prot['Nb_interactions_per_prot']:
        if val==1:
            category.append('1')
        elif 2 <= val <=4:
            category.append('[2,4]')
        elif 5 <= val <= 10:
            category.append('[5,10]')
        elif 11 <= val <= 20:
            category.append('[11,20]')
        elif 21 <= val <= 30:
            category.append('[21,30]')
        else:
            category.append('> 30')

    final_nb_interactions_per_prot['category_per_prot']=category

    all_predictions_w_cat = pd.merge(all_predictions, 
                                     final_nb_interactions_per_prot[['UniProt ID', 'category_per_prot']], 
                                     on='UniProt ID', 
                                     how="left")

    all_predictions_w_cat.to_csv(pred_stats_dirname + pattern_name + "_all_predictions_w_cat_20201113.csv") 