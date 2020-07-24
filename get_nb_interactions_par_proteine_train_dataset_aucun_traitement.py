import csv
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn as sns
import sys

sys.path.append('..')

root = '../CFTR_PROJECT/'

DB_version = "drugbank_v5.1.5"
DB_type = "S0h"

from DTI_prediction.process_dataset.process_DB import get_DB
DB = get_DB(DB_version, DB_type)

# 1 - VMO

process_name = "VMO"

# pattern_name variable
pattern_name = DB_type + '_' + process_name
# data_dir variable 
data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name + '/'

# # Liste des train datasets

clf_dirname = root + data_dir + 'classifiers/kronSVM/'
train_datasets_filename = clf_dirname + pattern_name + \
    '_kronSVM_list_train_datasets_20200619.data'
list_train_datasets = pickle.load(open(train_datasets_filename, 'rb'))

mean_nb_true_interactions_per_prot_list = []
mean_nb_false_interactions_per_prot_list = []

for prot_id in list(DB.proteins.dict_ind2prot.values()):
    
    nb_true_interactions_per_prot_list = []
    nb_false_interactions_per_prot_list = []

    for i in range(5):
        train_dataset = pd.DataFrame(list_train_datasets[i], columns=['UniProt ID', 'DrugbankID', 'interaction_bool'])

        true_interactions = train_dataset[train_dataset['interaction_bool']=='1']
        false_interactions = train_dataset[train_dataset['interaction_bool']=='0']

        prot_true_interactions = true_interactions[true_interactions['UniProt ID']==prot_id]
        prot_false_interactions = false_interactions[false_interactions['UniProt ID']==prot_id]

        nb_true_interactions_per_prot_list.append(prot_true_interactions.shape[0])
        nb_false_interactions_per_prot_list.append(prot_false_interactions.shape[0])

    mean_nb_true_interactions_per_prot_list.append(np.average(nb_true_interactions_per_prot_list))
    mean_nb_false_interactions_per_prot_list.append(np.average(nb_false_interactions_per_prot_list))

nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(DB.proteins.dict_ind2prot.values()),
                                        'Nb_true_interactions':mean_nb_true_interactions_per_prot_list,
                                        'Nb_false_interactions':mean_nb_false_interactions_per_prot_list})

nb_interactions_per_prot = nb_interactions_per_prot.sort_values(by=['Nb_true_interactions'], ascending=False)

raw_data_dir = 'data/' + DB_version + '/raw/'
raw_df = pd.read_csv(root + raw_data_dir + \
                    'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                    sep=",")
raw_df = raw_df.fillna('')

final_nb_interactions = pd.merge(nb_interactions_per_prot,
                                raw_df[['UniProt ID', 'Gene Name', 'Name']],
                                left_on='UniProt ID',
                                right_on='UniProt ID')

final_nb_interactions = final_nb_interactions.drop_duplicates()

final_nb_interactions.to_csv("nombre_interactions_par_proteine_train_dataset_aucun_traitement_20200619.csv")

final_nb_interactions = pd.read_csv('nombre_interactions_par_proteine_train_dataset_aucun_traitement_20200619.csv', encoding='utf-8')

true_dist = sns.distplot(final_nb_interactions['Nb_true_interactions'])
plt.savefig('dist_true_interactions_par_proteine_train_dataset_aucun_traitement_20200619.png')
plt.close()

false_dist = sns.distplot(final_nb_interactions['Nb_false_interactions'], )
plt.savefig('dist_false_interactions_par_proteine_train_dataset_aucun_traitement_20200619.png')
plt.close()