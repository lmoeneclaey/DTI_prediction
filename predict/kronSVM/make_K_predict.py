import numpy as np

from process_dataset.DB_utils import Drugs, Proteins, FormattedDB

root = './../CFTR_PROJECT/'

def get_list_couples_predict_mol(dbid, preprocessed_DB):

    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot

    list_couples_predict = []
    for key in range(len(dict_ind2prot)):
        list_couples_predict.append((dict_ind2prot[key],dbid))

    return list_couples_predict

def get_list_couples_predict_prot(dbid, preprocessed_DB):

    dict_ind2mol = preprocessed_DB[1]

    list_couples_predict = []
    for key in range(len(dict_ind2mol)):
        list_couples_predict.append((dbid,dict_ind2mol[key]))

    return list_couples_predict

def make_K_predict(dbid, preprocessed_DB, kernels, list_couples_train):

    # get the preprocessed DBdatabase 
    # preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_mol2ind = preprocessed_DB[2]
    dict_prot2ind = preprocessed_DB[5]

    # couples in train
    nb_couples_train = len(list_couples_train)

    # couples for prediction
    if dbid[:2] == 'DB':
        list_couples_predict = get_list_couples_predict_mol(dbid,preprocessed_DB)
    else:
        list_couples_predict = get_list_couples_predict_prot(dbid,preprocessed_DB)
    nb_couples_predict = len(list_couples_predict)

    # get the kernels
    # kernels = get_K_mol_K_prot(DB_version, DB_type, process_name, norm_option)
    # no need anymore of the argument "norm_option"
    K_mol = kernels[0]
    K_prot = kernels[1]

    # process the similarity kernel
    K_predict = np.zeros((nb_couples_predict, nb_couples_train))
    for i in range(nb_couples_predict):
        ind1_prot = dict_prot2ind[list_couples_predict[i][0]]
        ind1_mol = dict_mol2ind[list_couples_predict[i][1]]
        for j in range(nb_couples_train):
            ind2_prot = dict_prot2ind[list_couples_train[j][0]]
            ind2_mol = dict_mol2ind[list_couples_train[j][1]]

            K_predict[i, j] = K_prot[ind1_prot, ind2_prot] * \
                K_mol[ind1_mol, ind2_mol]

    return K_predict