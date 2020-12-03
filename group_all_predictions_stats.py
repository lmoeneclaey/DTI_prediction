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

    # Ici il faut mettre un choix entre deux dates
    pred_stats_dirname = root + data_dir + '/predictions/kronSVM/'
    
    for i in range(DB.drugs.nb):

        dbid = DB.drugs.dict_ind2mol[i]

        ## Répartition des repartition par tranche de rangs
        rank_repartition_filename = "../" + pattern_name +'/rank_repartition/' + dbid + '_rank_repartition_20201113.csv'
        if os.path.exists(rank_repartition_filename):
            rank_repartition_per_drug_pd = pd.read_csv(rank_repartition_filename)   
            if i==0:
                all_drugs_rank_repartition_pd = copy.deepcopy(rank_repartition_per_drug_pd)
            else:
                all_drugs_rank_repartition_pd = pd.concat([all_drugs_rank_repartition_pd,
                                                          rank_repartition_per_drug_pd],
                                                          axis=0)

        ## Rang moyen de chaque catégorie pour les 100 premières
        # mean_rank_filename = '../' + pattern_name + '/mean_rank_per_cat/' + dbid + '_mean_rank_per_cat.csv'
        # if os.path.exists(mean_rank_filename):
        #     mean_rank_per_cat = pd.read_csv(mean_rank_filename)
        #     if i==0:
        #         all_drugs_mean_rank_pd = copy.deepcopy(mean_rank_per_cat)
        #     else:
        #         all_drugs_mean_rank_pd = pd.concat([all_drugs_mean_rank_pd,
        #                                             mean_rank_per_cat], 
        #                                             axis=0)

    all_drugs_rank_repartition_pd.to_csv(pred_stats_dirname + pattern_name + "_all_drugs_rank_repartition_20201113.csv")
    # all_drugs_mean_rank_pd.to_csv(pred_stats_dirname + pattern_name + "_all_drugs_mean_rank_per_cat.csv")