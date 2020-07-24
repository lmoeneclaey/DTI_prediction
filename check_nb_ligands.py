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

# raw_data_dir = 'data/' + DB_version + '/raw/'
# raw_df = pd.read_csv(root + raw_data_dir + \
#                     'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
#                     sep=",")
# raw_df = raw_df.fillna('')

# # 1. Histogramme des interactions

# interactions_pd = pd.DataFrame(DB.interactions.array, columns=['UniProt ID', 'DrugbankID', 'interaction_bool'])

# interactions_pd

# nb_interactions_per_prot_list = []
# for prot_id in list(DB.proteins.dict_ind2prot.values()):
#     prot_interactions = interactions_pd[interactions_pd['UniProt ID']==prot_id]
#     nb_interactions_per_prot_list.append(prot_interactions.shape[0])

# nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(DB.proteins.dict_ind2prot.values()),
#                                         'Nb_interactions':nb_interactions_per_prot_list})

# nb_interactions_per_prot = nb_interactions_per_prot.sort_values(by=['Nb_interactions'], ascending=False)

# final_nb_interactions = pd.merge(nb_interactions_per_prot,
#                                 raw_df[['UniProt ID', 'Gene Name', 'Name']],
#                                 left_on='UniProt ID',
#                                 right_on='UniProt ID')

# final_nb_interactions = final_nb_interactions.drop_duplicates()

# final_nb_interactions.to_csv("nombre_ligands_par_proteine_DrugBank_20200715.csv")

nb_interactions = pd.read_csv('nombre_ligands_par_proteine_DrugBank_20200715.csv', encoding='utf-8')

# violin = sns.violinplot(final_nb_interactions['Nb_interactions'])
# plt.savefig('nb_interactions_per_protein_violin.png')
# plt.close()

# print('moy:', np.average(final_nb_interactions['Nb_interactions']))
# print('min:', np.min(final_nb_interactions['Nb_interactions']))
# print('max:', np.max(final_nb_interactions['Nb_interactions']))

top_50 = nb_interactions[0:50]

sns.set_context(rc={"font.size":5})
top_50 = sns.barplot(x='Gene Name', y="Nb_interactions", data= top_50)
plt.xticks(rotation=90)
plt.savefig('top_50_nb_interactions_per_protein.png', dpi = 300)
plt.close()