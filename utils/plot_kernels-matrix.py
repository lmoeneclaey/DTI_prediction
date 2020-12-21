import csv
import copy
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys

from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, check_protein, check_couple, get_couples_from_array

root = '../CFTR_PROJECT/'

DB_version = "drugbank_v5.1.5"
DB_type = "S0h"
process_name = "VMO"

# pattern_name variable
pattern_name = DB_type + '_' + process_name
# data_dir variable 
data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name + '/'

preprocessed_DB = get_DB(DB_version, DB_type)

from DTI_prediction.process_dataset.correct_interactions import get_orphan

ivacaftor_dbid = 'DB08820'
corrected_DB = copy.deepcopy(preprocessed_DB)
corrected_DB = get_orphan(DB=corrected_DB, dbid=ivacaftor_dbid)

# Get the kernels 

from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot
from DTI_prediction.make_kernels.make_K_mol import center_and_normalise_kernel

kernels = get_K_mol_K_prot(DB_version, DB_type, norm=False)
K_mol = kernels[0]
K_prot = kernels[1]

from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset, get_train_dataset, make_K_train

train_dataset = get_train_dataset(2821, corrected_DB)
K_train = make_K_train(train_dataset, corrected_DB, kernels)

K_train_norm = center_and_normalise_kernel(K_train)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# # Plot K_mol heat map test

# # heatmap + color map
# plt.pcolor(K_mol, cmap=mpl.cm.PuRd)

# # plot colorbar to the right
# plt.colorbar()

# # set axes boundaries
# plt.xlim([0, K_mol.shape[0]])
# plt.ylim([0, K_mol.shape[0]])

# # flip the y-axis
# plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()

# plt.savefig(root + data_dir + 'K_mol.png', dpi = 300)
# plt.close()

# # Plot K_prot heat map test

# # heatmap + color map
# plt.pcolor(K_prot, cmap=mpl.cm.PuRd)

# # plot colorbar to the right
# plt.colorbar()

# # set axes boundaries
# plt.xlim([0, K_prot.shape[0]])
# plt.ylim([0, K_prot.shape[0]])

# # flip the y-axis
# plt.gca().invert_yaxis()
# plt.gca().xaxis.tick_top()

# plt.savefig(root + data_dir + 'K_prot.png', dpi = 300)
# plt.close()

# Plot K_train_norm heat map test

K_test_norm = K_train_norm[:1000, :1000]

# heatmap + color map
plt.pcolor(K_test_norm, cmap=mpl.cm.PuRd)

# plot colorbar to the right
plt.colorbar()

# set axes boundaries
plt.xlim([0, K_test_norm.shape[0]])
plt.ylim([0, K_test_norm.shape[0]])

# flip the y-axis
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()

plt.savefig(root + data_dir + 'K_train_K_mol_K_prot_unnorm_centree_reduite.png', dpi = 300)
plt.close()

K_test = K_train[:1000, :1000]

# heatmap + color map
plt.pcolor(K_test, cmap=mpl.cm.PuRd)

# plot colorbar to the right
plt.colorbar()

# set axes boundaries
plt.xlim([0, K_test.shape[0]])
plt.ylim([0, K_test.shape[0]])

# flip the y-axis
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()

plt.savefig(root + data_dir + 'K_train_K_mol_K_prot_unnorm.png', dpi = 300)
plt.close()