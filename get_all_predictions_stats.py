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

    parser.add_argument("dbid", type=str,
                        help = "the DrugBankId of the molecule/protein of which\
                        we want to predict the interactions")

    args = parser.parse_args()

    DB = get_DB(args.DB_version, args.DB_type)

    # pattern_name variable
    pattern_name = args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name + '/'

    # Load train datasets
    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'
    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'
    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    train_dataset_pd = pd.DataFrame(train_datasets_array[0], columns=['UniProt ID', 'DrugBank ID', 'interaction_bool'])
    train_dataset_pd['interaction_bool'] = train_dataset_pd['interaction_bool'].astype(int)

    # Ici il faut mettre un choix entre deux dates
    pred_dirname = "../" + pattern_name + '/predictions/kronSVM/'
    
    # dict_cat = {0:'0', 1:'[1,4]', 2:'[5,10]', 3:'> 10'}
    dict_cat = {0:'1', 1:'[2,4]', 2:'[5,10]', 3:'[11,20]', 4:'[21,30]', 5:'> 30'}
    
    ranks_split = []
    for isplit in range(26):
        ranks_split.append(str(isplit*100) + " - " + str((isplit+1)*100-1))

    all_drugs_mean_rank_list = []
    
    predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + args.dbid + '_pred_clean_20201031.csv'
    if not(os.path.exists(predictions_filename)):
        predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + args.dbid + '_pred_clean_20201101.csv'
        if not(os.path.exists(predictions_filename)):
            predictions_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + args.dbid + '_pred_clean_20201102.csv'
    predictions = pd.read_csv(predictions_filename, encoding='utf-8')
    
    final_predictions = get_number_of_interactions_per_prot(train_dataset_pd=train_dataset_pd,
                                                            test_dataset_pd=predictions)
    
    all_cat_count = dict(collections.Counter(final_predictions.category_prot))
    
    nb_interactions_in_train = train_dataset_pd[(train_dataset_pd['DrugBank ID']==args.dbid) &
                                                (train_dataset_pd['interaction_bool']==1)].shape[0]

    if nb_interactions_in_train==0:
        category_drug = '0'
    elif 1 <= nb_interactions_in_train <=4:
        category_drug = '[1,4]'
    elif 5 <= nb_interactions_in_train <= 10:
        category_drug = '[5,10]'
    else:
        category_drug = '> 10'
    
    ## Répartition des repartition par tranche de rangs
    rank_repartition_per_drug_cat0 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
    rank_repartition_per_drug_cat1 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
    rank_repartition_per_drug_cat2 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
    rank_repartition_per_drug_cat3 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
    rank_repartition_per_drug_cat4 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
    rank_repartition_per_drug_cat5 = pd.DataFrame(columns=["DrugBankID","category_prot","category_drug","nb","ratio"], 
                                                index=ranks_split)
                                                
    
    rank_repartition_per_drug_all_cats = [rank_repartition_per_drug_cat0,
                                        rank_repartition_per_drug_cat1,
                                        rank_repartition_per_drug_cat2,
                                        rank_repartition_per_drug_cat3,
                                        rank_repartition_per_drug_cat4,
                                        rank_repartition_per_drug_cat5]
    
    for isplit in range(26):
        
        df = final_predictions[isplit*100:(isplit+1)*100]

        cat_count = dict(collections.Counter(df.category_prot))
        
        for icat in range(6):

            if not(dict_cat[icat] in cat_count.keys()):
                cat_count[dict_cat[icat]]=0
                
            rank_repartition_per_drug_all_cats[icat].loc[ranks_split[isplit]] = [args.dbid, 
                                                                                dict_cat[icat],
                                                                                category_drug,
                                                                                cat_count[dict_cat[icat]],
                                                                                cat_count[dict_cat[icat]]*100/all_cat_count[dict_cat[icat]]]
    
    for icat in range(6):
        rank_repartition_per_drug_all_cats[icat]['rank_slice'] = ranks_split
        rank_repartition_per_drug_all_cats[icat].reset_index()
        
    rank_repartition_per_drug_pd = pd.concat(rank_repartition_per_drug_all_cats, axis=0) 
    rank_repartition_per_drug_pd.to_csv("../" + pattern_name +'/rank_repartition/' + args.dbid + '_rank_repartition_20201113.csv')

    ## Rang moyen de chaque catégorie pour les 100 premières

    # df = final_predictions[0:100]
    # df = df.drop(columns='Unnamed: 0')
    # df['rank'] = df.index

    # mean_rank_per_cat_list = [category_drug]
    # for icat in np.arange(1,4):
    #     mean_rank_per_cat_list.append(df[df['category_prot']==dict_cat[icat]]["rank"].mean())
    # mean_rank_per_cat_pd = pd.DataFrame(np.array(mean_rank_per_cat_list).reshape(-1,4), columns=["category_drug", 
    #                                                                                              "mean_rank_cat1", 
    #                                                                                              "mean_rank_cat2", 
    #                                                                                              "mean_rank_cat3"])
    
    # all_drugs_mean_rank_list.append(mean_rank_per_cat_list)

    ## Répartition des repartition par tranche de rangs

    ## Rang moyen de chaque catégorie pour les 100 premières
    # changer aussi pour toutes les drugs
    # mean_rank_per_cat_pd = pd.DataFrame(np.vstack(all_drugs_mean_rank_list), 
    #                                 columns = ["category_drug", "mean_rank_cat1", "mean_rank_cat2", "mean_rank_cat3"],
    #                                 index = list(DB.drugs.dict_ind2mol.values())[:50])
    # mean_rank_per_cat_pd['DrugBank ID'] = args.dbid
    # mean_rank_per_cat_pd.reset_index()
    # mean_rank_per_cat_pd.to_csv("../" + pattern_name + '/mean_rank_per_cat/' + args.dbid + '_mean_rank_per_cat.csv')

    




