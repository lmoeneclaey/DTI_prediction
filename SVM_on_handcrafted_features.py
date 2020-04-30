import pickle

def get_Kmol_Kprot(DB_version, DB_type, process_name, norm_option):
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
    norm_option : str
        normalized or unnormalized

    Returns
    -------
    Kmol
    Kprot
    """   

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    if norm_option == 'unnorm':
        K_mol = pickle.load(open(root + data_dir + pattern_name + '_Kmol.data', 'rb'))
        K_prot = pickle.load(open(root + data_dir + pattern_name + '_Kprot.data', 'rb'))
    elif norm_option == 'norm':
        K_mol = pickle.load(open(root + data_dir + pattern_name + '_Kmol_norm.data', 'rb'))
        K_prot = pickle.load(open(root + data_dir + pattern_name + '_Kprot_norm.data', 'rb'))

    return Kmol, Kprot

def make_Kinter(list_couples, Kprot, Kmol, dict_prot2ind, dict_mol2ind):
    """ 
    Process the kernels of interactions based on the list of interactions in \
        the data set, the list of proteins and molecules, and the kernels of \
        proteins and molecules.  

    Parameters
    ----------
    list_couples : list
    Kprot : numpy array
    Kmol : numpy array
    dict_prot2ind : dictionary
    dict_mol2ind : dictionary

    Returns
    -------
    None
    """

    K = np.zeros((nb_couple, nb_couple))
    for i in range(nb_couple):
        ind1_prot = dict_prot2ind[list_couple[i][0]]
        ind1_mol = dict_mol2ind[list_couple[i][1]]
        for j in range(i, nb_couple):
            ind2_prot = dict_prot2ind[list_couple[j][0]]
            ind2_mol = dict_mol2ind[list_couple[j][1]]

            K[i, j] = K_mol[ind1_mol, ind2_mol] * K_prot[ind1_prot, ind2_prot]
            K[j, i] = K[i, j]



def make_Kcouple(x1, x2, Kmol, Kprot, dict_prot2ind, dict_mol2ind):

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
                K_temp[i, j] = Kmol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    Kprot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
                K_temp[j, i] = K_temp[i, j]
        else:  # if it is for validation or test data
            for j in range(len(x2)):
                prot2, mol2 = x2[j]
                K_temp[i, j] = Kmol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    Kprot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
    # in the case of train data, K_temp is "nb_sample_in_train * nb_sample_in_train"
    # in the case of test/val data, K_temp is "nb_sample_in_train * nb_sample_in_test/val"
    return K_temp

    