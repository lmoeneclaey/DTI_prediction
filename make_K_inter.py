import numpy as np
import pickle

root = "./../CFTR_PROJECT/"

def get_K_mol_K_prot(DB_version, DB_type, process_name, norm):
    """ 
    Load the molecules and the proteins kernels

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'
    norm : boolean
        normalized or unnormalized
        "normalized" by default

    Returns
    -------
    K_mol
    K_prot
    """   

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    if norm == False:
        K_mol = pickle.load(open(root + data_dir + pattern_name + '_K_mol.data', 'rb'))
        K_prot = pickle.load(open(root + data_dir + pattern_name + '_K_prot.data', 'rb'))
    elif norm == True:
        K_mol = pickle.load(open(root + data_dir + pattern_name + '_K_mol_norm.data', 'rb'))
        K_prot = pickle.load(open(root + data_dir + pattern_name + '_K_prot_norm.data', 'rb'))

    return K_mol, K_prot

def make_Kcouple(x1, x2, K_mol, K_prot, dict_prot2ind, dict_mol2ind):

    # x1 is the list of (prot, mol) IDs in the train data
    # x2 is the list of (prot, mol) IDs in the validation or test data
    if x2 is None:  # if it is for train data
        K_temp = np.zeros((len(x1), len(x1)))
    else:  # if it is for validation or test data
        K_temp = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        prot1, mol1 = x1[i]
        if x2 is None:  # if it is for train data
            for j in range(i, len(x1)):
                prot2, mol2 = x1[j]
                K_temp[i, j] = K_mol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    K_prot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
                K_temp[j, i] = K_temp[i, j]
        else:  # if it is for validation or test data
            for j in range(len(x2)):
                prot2, mol2 = x2[j]
                K_temp[i, j] = K_mol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    K_prot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
    # in the case of train data, K_temp is "nb_sample_in_train * nb_sample_in_train"
    # in the case of test/val data, K_temp is "nb_sample_in_train * nb_sample_in_test/val"
    return K_temp  