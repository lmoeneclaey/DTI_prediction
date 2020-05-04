import numpy as np

from process_dataset.process_DB import get_DB
from make_K_inter import get_K_mol_K_prot

root = './../CFTR_PROJECT/'

def forbid_couple(couple, preprocessed_DB):
    """
    Assign 0 to a couple in the matrix of interactions

    Parameters
    ----------
    couple : tuple 
        (prot:DrugBankID, mol:DrugBankID)
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    preprocessed_DB
        where intMat has been changed
    """ 
    dict_mol2ind = preprocessed_DB[2]
    dict_prot2ind = preprocessed_DB[5]
    intMat = preprocessed_DB[6]

    if intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] == 0:
        print('attention : forbidden couple ' + str(couple) + ' is already neg')
    else:
        intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] = 0
    
    preprocessed_DB[6] = intMat

    return preprocessed_DB

def get_list_couples_train(seed, preprocessed_DB):
    """ 
    Get the list of all the couples that are in the train:
        - the "true" (known) interactions (with indices ind_true_inter)
        _ the "false" (unknown) interactions (with indices ind_false_inter)  

    Parameters
    ----------
    seed : number
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    list_couples : list
        List of all the couples in the train data set
    y : list
        results to predict
    ind_true_inter : np.array
    ind_false_inter : np.array
    """ 

    dict_ind2mol = preprocessed_DB[1]
    dict_ind2prot = preprocessed_DB[4]
    intMat = preprocessed_DB[6]
    list_interactions = preprocessed_DB[7]
    print("0:", len(list_interactions))

    # Set the seed
    np.random.seed(seed)

    # get the interactions indices
    # ind_true_inter : indices where there is an interaction
    # ind_false_inter : indices where there is not an interaction
    ind_true_inter = np.where(intMat == 1) 
    nb_true_inter = len(list_interactions)
    all_ind_false_inter = np.where(intMat == 0)
    nb_false_inter = len(all_ind_false_inter[0])

    # choose between all the "false" interactions, nb_true_interactions couples,
    # without replacement
    mask = np.random.choice(np.arange(nb_false_inter), 
                            nb_true_inter,
                            replace=False)

    # get a new list with only the "false" interactions indices which will be \
    # in the train data set
    ind_false_inter = (all_ind_false_inter[0][mask], 
                       all_ind_false_inter[1][mask])

    # initialisation
    list_couples, y = [], []

    # true interactions
    for i in range(nb_true_inter):
        list_couples.append((dict_ind2prot[ind_true_inter[0][i]], 
                             dict_ind2mol[ind_true_inter[1][i]]))
        y.append(1)
    print("1:", len(list_couples))

    # the nb of "false" interactions in the set of couples is equal to the \
    # number of true interactions 
    for i in range(nb_true_inter):
        list_couples.append((dict_ind2prot[ind_false_inter[0][i]],
                            dict_ind2mol[ind_false_inter[1][i]]))
        y.append(0)

    y = np.array(y)

    print("list of all the couples done")
    return list_couples, y, ind_true_inter, ind_false_inter

def make_K_train(seed, preprocessed_DB, kernels, forbidden_list=None):
    """ 
    Compute the interactions kernels for the train data set

    Calculate them based on the kernels on proteins and on molecules.

    Use center_and_normalise_kernel()

    Parameters
    ----------
    seed : int
        seed for the "false" interactions in the train data set
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'
    forbidden_list : list of tuples
        if you want to forbid some couples as "true" interactions

    Returns
    -------
    K_train
    """

    # get the preprocessed DBdatabase 
    # preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_mol2ind = preprocessed_DB[2]
    dict_prot2ind = preprocessed_DB[5]

    # forbid some couples
    if forbidden_list is not None:
        for couple in forbidden_list:
            corrected_preprocessed_DB = forbid_couple(couple, preprocessed_DB)
            preprocessed_DB = corrected_preprocessed_DB

    # get the train dataset
    train_set = get_list_couples_train(seed,
                                      preprocessed_DB)
    list_couples = train_set[0]
    nb_couples = len(list_couples)

    # get the kernels
    # kernels = get_K_mol_K_prot(DB_version, DB_type, process_name, norm_option)
    # no need anymore of the argument "norm_option"
    K_mol = kernels[0]
    K_prot = kernels[1]

    # process the kernels of interactions
    K_train = np.zeros((nb_couples, nb_couples))
    for i in range(nb_couples):
        ind1_prot = dict_prot2ind[list_couples[i][0]]
        ind1_mol = dict_mol2ind[list_couples[i][1]]
        for j in range(i, nb_couples):
            ind2_prot = dict_prot2ind[list_couples[j][0]]
            ind2_mol = dict_mol2ind[list_couples[j][1]]

            K_train[i, j] = K_prot[ind1_prot, ind2_prot] * \
                K_mol[ind1_mol, ind2_mol]
            K_train[j, i] = K_train[i, j]

    return K_train