import numpy as np

root = './../CFTR_PROJECT/'

def make_K_test(list_couples_train, list_couples_test, preprocessed_DB, kernels):

    # get the preprocessed DBdatabase 
    # preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_mol2ind = preprocessed_DB[2]
    dict_prot2ind = preprocessed_DB[5]

    # couples in train
    nb_couples_train = len(list_couples_train)

    # couples for prediction
    nb_couples_test = len(list_couples_test)

    # get the kernels
    # kernels = get_K_mol_K_prot(DB_version, DB_type, process_name, norm_option)
    # no need anymore of the argument "norm_option"
    K_mol = kernels[0]
    K_prot = kernels[1]

    # process the similarity kernel
    K_test = np.zeros((nb_couples_test, nb_couples_train))
    for i in range(nb_couples_test):
        ind1_prot = dict_prot2ind[list_couples_test[i][0]]
        ind1_mol = dict_mol2ind[list_couples_test[i][1]]
        for j in range(nb_couples_train):
            ind2_prot = dict_prot2ind[list_couples_train[j][0]]
            ind2_mol = dict_mol2ind[list_couples_train[j][1]]

            K_test[i, j] = K_prot[ind1_prot, ind2_prot] * \
                K_mol[ind1_mol, ind2_mol]

    return K_test