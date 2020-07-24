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

nb_ligands_kernel_dir = "../../Thèse/Chemogenomics/Matthieu/Nb_ligands_kernels/"

root = '../CFTR_PROJECT/'

DB_version = "drugbank_v5.1.5"
DB_type = "S0h"

from DTI_prediction.process_dataset.process_DB import get_DB
DB = get_DB(DB_version, DB_type)

# Nb de ligands par protéines

final_nb_interactions = pd.read_csv('nombre_interactions_par_proteine_train_dataset_tout_centre_norme_20200609.csv', encoding='utf-8')

# 1 - VMO

process_name = "VMO"

# pattern_name variable
pattern_name = DB_type + '_' + process_name
# data_dir variable 
data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name + '/'

# C - Tout centré normé - 2020/06/09

pred_dirname = root + data_dir + 'predictions/20200609/'

ivacaftor_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_norm_' + 'DB08820' + '_pred_clean_20200609.csv', encoding='utf-8')
tezacaftor_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_norm_' + 'DB11712' + '_pred_clean_20200609.csv', encoding='utf-8')
lumacaftor_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_norm_' + 'DB09280' + '_pred_clean_20200609.csv', encoding='utf-8')
elexacaftor_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_norm_' + 'DB15444' + '_pred_clean_20200609.csv', encoding='utf-8')

# B - 1 Ivacaftor

ivacaftor_proteins_nb_true_interactions = pd.merge(ivacaftor_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_true_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

ivacaftor_proteins_nb_false_interactions = pd.merge(ivacaftor_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_false_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

df_true = pd.DataFrame({'Gene Name' : ivacaftor_proteins_nb_true_interactions['Gene Name'],
                        'nb_interactions': ivacaftor_proteins_nb_true_interactions['Nb_true_interactions'],
                        'type': 'True'})

df_false = pd.DataFrame({'Gene Name' : ivacaftor_proteins_nb_false_interactions['Gene Name'],
                        'nb_interactions': ivacaftor_proteins_nb_false_interactions['Nb_false_interactions'],
                        'type': 'False'})

top_50 = pd.concat([df_true[:50],df_false[:50]])

sns.set_context(rc={"font.size":5})
top_50_barplot_ivacaftor_predictions_nb_ligands = sns.barplot(x='Gene Name', y="nb_interactions", hue="type", data= top_50)
plt.xticks(rotation=90)
plt.savefig('top_50_nb_interactions_per_protein_predicted_for_ivacaftor_tout_centre_norme_20200609.png', dpi = 300)
plt.close()

sns.set_context(rc={"font.size":5})
ivacaftor_predictions_nb_ligands = sns.barplot(x='Gene Name', y="nb_interactions", data= df_true, color="blue")
ivacaftor_predictions_nb_ligands.axes.get_xaxis().set_visible(False)
plt.savefig('nb_ligands_per_protein_predicted_for_ivacaftor_tout_centre_norme_20200609.png', dpi = 300)
plt.close()

# B - 2 Lumacaftor

lumacaftor_proteins_nb_true_interactions = pd.merge(lumacaftor_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_true_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

lumacaftor_proteins_nb_false_interactions = pd.merge(lumacaftor_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_false_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

df_true = pd.DataFrame({'Gene Name' : lumacaftor_proteins_nb_true_interactions['Gene Name'],
                        'nb_interactions': lumacaftor_proteins_nb_true_interactions['Nb_true_interactions'],
                        'type': 'True'})

df_false = pd.DataFrame({'Gene Name' : lumacaftor_proteins_nb_false_interactions['Gene Name'],
                        'nb_interactions': lumacaftor_proteins_nb_false_interactions['Nb_false_interactions'],
                        'type': 'False'})

top_50 = pd.concat([df_true[:50],df_false[:50]])

sns.set_context(rc={"font.size":5})
top_50_barplot_lumacaftor_predictions_nb_ligands = sns.barplot(x='Gene Name', y="nb_interactions", hue="type", data= top_50)
plt.xticks(rotation=90)
plt.savefig('top_50_nb_interactions_per_protein_predicted_for_lumacaftor_tout_centre_norme_20200609.png', dpi = 300)
plt.close()

sns.set_context(rc={"font.size":5})
lumacaftor_predictions_nb_ligands = sns.barplot(x='Gene Name', y="nb_interactions", data= df_true, color="blue")
lumacaftor_predictions_nb_ligands.axes.get_xaxis().set_visible(False)
plt.savefig('nb_ligands_per_protein_predicted_for_lumacaftor_tout_centre_norme_20200609.png', dpi = 300)
plt.close()

# # B - 1 Ivacaftor

# ivacaftor_proteins_nb_ligands = pd.merge(ivacaftor_predictions,
#                                          nb_interactions[['UniProt ID', "Nb_interactions"]],
#                                          left_on='UniProt ID',
#                                          right_on='UniProt ID')

# print(ivacaftor_proteins_nb_ligands[:30])

# top_100 = ivacaftor_proteins_nb_ligands[:100]

# plt.rcParams.update({'font.size':5})
# plt.xticks(rotation=90)

# barplot_ivacaftor_predictions_nb_ligands = sns.barplot(x=top_100['Gene Name'], y=top_100['Nb_interactions'])
# plt.savefig(nb_ligands_kernel_dir + 'nb_ligands_per_protein_predicted_for_ivacaftor_tout_centre_norme_20200609.png')
# plt.close()

# # B - 2 Lumacaftor

# lumacaftor_proteins_nb_ligands = pd.merge(lumacaftor_predictions,
#                                          nb_interactions[['UniProt ID', "Nb_interactions"]],
#                                          left_on='UniProt ID',
#                                          right_on='UniProt ID')

# print(lumacaftor_proteins_nb_ligands[:30])

# top_100 = lumacaftor_proteins_nb_ligands[:100]

# plt.rcParams.update({'font.size':5})
# plt.xticks(rotation=90)

# barplot_lumacaftor_predictions_nb_ligands = sns.barplot(x=top_100['Gene Name'], y=top_100['Nb_interactions'])
# plt.savefig(nb_ligands_kernel_dir + 'nb_ligands_per_protein_predicted_for_lumacaftor_kernels_tout_centre_norme_20200609.png')
# plt.close()

# # B - 3 Tezacaftor

# tezacaftor_proteins_nb_ligands = pd.merge(tezacaftor_predictions,
#                                          nb_interactions[['UniProt ID', "Nb_interactions"]],
#                                          left_on='UniProt ID',
#                                          right_on='UniProt ID')

# print(tezacaftor_proteins_nb_ligands[:30])

# top_100 = tezacaftor_proteins_nb_ligands[:100]

# plt.rcParams.update({'font.size':5})
# plt.xticks(rotation=90)

# barplot_tezacaftor_predictions_nb_ligands = sns.barplot(x=top_100['Gene Name'], y=top_100['Nb_interactions'])
# plt.savefig(nb_ligands_kernel_dir + 'nb_ligands_per_protein_predicted_for_tezacaftor_kernels_tout_centre_norme_20200609.png')
# plt.close()

# # B - 4 Elexacaftor

# elexacaftor_proteins_nb_ligands = pd.merge(elexacaftor_predictions,
#                                          nb_interactions[['UniProt ID', "Nb_interactions"]],
#                                          left_on='UniProt ID',
#                                          right_on='UniProt ID')

# print(elexacaftor_proteins_nb_ligands[:30])

# top_100 = elexacaftor_proteins_nb_ligands[:100]

# plt.rcParams.update({'font.size':5})
# plt.xticks(rotation=90)

# barplot_elexacaftor_predictions_nb_ligands = sns.barplot(x=top_100['Gene Name'], y=top_100['Nb_interactions'])
# plt.savefig(nb_ligands_kernel_dir + 'nb_ligands_per_protein_predicted_for_elexacaftor_kernels_tout_centre_norme_20200609.png')
# plt.close()