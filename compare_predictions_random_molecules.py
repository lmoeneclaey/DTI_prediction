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

# Nb de ligands par protéines

final_nb_interactions = pd.read_csv('nombre_ligands_par_proteine_DrugBank_20200715.csv', encoding='utf-8')

# 1 - atovaquone

process_name = "atovaquone-orphan"

# pattern_name variable
pattern_name = DB_type + '_' + process_name
# data_dir variable 
data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name + '/'

pred_dirname = root + data_dir + 'predictions/20200609/'

Kprot_norme_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_centered_norm_DB01117_pred_clean_20200717.csv', encoding='utf-8')
tout_centre_norme_predictions = pd.read_csv(pred_dirname + pattern_name + '_kronSVM_norm_DB01117_pred_clean_20200709.csv', encoding='utf-8')

# B - 1 Ivacaftor

Kprot_norme_predictions_nb_interactions = pd.merge(Kprot_norme_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

tout_centre_norme_predictions_nb_interactions = pd.merge(tout_centre_norme_predictions,
                                                    final_nb_interactions[['UniProt ID', 'Nb_interactions']],
                                                    left_on='UniProt ID',
                                                    right_on='UniProt ID')

# df_true = pd.DataFrame({'Gene Name' : ivacaftor_proteins_nb_true_interactions['Gene Name'],
#                         'nb_interactions': ivacaftor_proteins_nb_true_interactions['Nb_true_interactions'],
#                         'type': 'True'})

# df_false = pd.DataFrame({'Gene Name' : ivacaftor_proteins_nb_false_interactions['Gene Name'],
#                         'nb_interactions': ivacaftor_proteins_nb_false_interactions['Nb_false_interactions'],
#                         'type': 'False'})

top_50_a = Kprot_norme_predictions_nb_interactions[:50]
top_50_b = tout_centre_norme_predictions_nb_interactions[:50]

sns.set_context(rc={"font.size":5})
top_50_Kprot_norme_predictions_nb_interactions = sns.barplot(x='Gene Name', y="Nb_interactions", data= top_50_a)
plt.xticks(rotation=90)
plt.savefig('top_50_nb_interactions_per_protein_predicted_for_atovaquone_Kprot_norme_20200709.png', dpi = 300)
plt.close()

sns.set_context(rc={"font.size":5})
top_50_tout_centre_norme_predictions_nb_interactions = sns.barplot(x='Gene Name', y="Nb_interactions", data= top_50_a)
plt.xticks(rotation=90)
plt.savefig('top_50_nb_interactions_per_protein_predicted_for_atovaquone_tout_centre_norme_20200717.png', dpi = 300)
plt.close()