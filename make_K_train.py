import numpy as np

from process_dataset.DB_utils import ListInteractions
# from process_dataset.process_DB import get_DB
# from make_K_inter import get_K_mol_K_prot

root = './../CFTR_PROJECT/'

class InteractionsTrainDataset:
    """
    Class definining a train data set including a number of "true" interactions
    and the exact same number of "false" interactions.
    So it is a list of two 'ListInteractions' objects.
    """

    def __init__(self, true_inter, false_inter):
        self.true_inter = true_inter
        self.false_inter = false_inter

    # we should write a function that verifies that both true_inter and \
    # false_inter are from the 'ListInteractions' class

def correct_unproven_interactions(interaction, preprocessed_DB):
    """
    Correct 1 to 0 in the matrix of interactions, interactions that haven't \
    been proven experimentally.

    Parameters
    ----------
    interaction : tuple of length 2
        (UniprotID, DrugbankID)
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    corrected_preprocessed_DB : tuple of length 8 
    """



# def get_list_couples_train(seed, preprocessed_DB):
def get_train_dataset(seed, preprocessed_DB):
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
    train_dataset : InteractionsTrainDataset
        List of all the "true" interactions in a 'ListInteractions' class and \
        all the "false" interactions, also in a 'ListInteractions' class. 
    true_inter : ListInteractions
        List of all the "true" interactions in the train dataset
    false_inter : ListInteractions
        List of all the "false" interactions in the train dataset
    """ 

    dict_ind2mol = preprocessed_DB[1]
    dict_ind2prot = preprocessed_DB[4]
    intMat = preprocessed_DB[6]
    list_interactions = preprocessed_DB[7]

    # TRUE INTERACTIONS
    # get the interactions indices
    # ind_true_inter : indices where there is an interaction
    ind_true_inter = np.where(intMat == 1) 
    nb_true_inter = len(list_interactions)

    true_inter = ListInteractions(list_couples=list_interactions,
                                  interaction_bool=np.array([1]*nb_true_inter),
                                  ind_inter=ind_true_inter)

    # "FALSE" INTERACTIONS
    # get the interactions indices
    # ind_false_inter : indices where there is not an interaction
    ind_all_false_inter = np.where(intMat == 0)
    nb_all_false_inter = len(ind_all_false_inter[0])

    # choose between all the "false" interactions, nb_true_interactions couples,
    # without replacement
    np.random.seed(seed)
    mask = np.random.choice(np.arange(nb_all_false_inter), 
                            nb_true_inter,
                            replace=False)

    # get a new list with only the "false" interactions indices which will be \
    # in the train dataset
    ind_false_inter = (ind_all_false_inter[0][mask], 
                       ind_all_false_inter[1][mask])
    nb_false_inter = len(ind_false_inter[0])

    list_false_inter = []
    for i in range(nb_false_inter):
        list_false_inter.append((dict_ind2prot[ind_false_inter[0][i]],
                                 dict_ind2mol[ind_false_inter[1][i]]))

    false_inter = ListInteractions(list_couples=list_false_inter,
                                   interaction_bool=np.array([0]*nb_false_inter),
                                   ind_inter=ind_false_inter)

    print("list of all the couples done.")

    # train_dataset = [true_inter, false_inter]
    train_dataset = InteractionsTrainDataset(true_inter=true_inter,
                                             false_inter=false_inter)
    print("Train dataset done.")

    # return true_inter, false_inter
    return train_dataset

# def make_K_train(seed, preprocessed_DB, kernels):
def make_K_train(train_dataset, preprocessed_DB, kernels):
    """ 
    Compute the interactions kernels for the train dataset

    Calculate them based on the kernels on proteins and on molecules.

    Use center_and_normalise_kernel()

    Parameters
    ----------
    seed : int
        seed for the "false" interactions in the train dataset
    train_dataset : InteractionsTrainDataset
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'

    Returns
    -------
    K_train
    """

    # get the preprocessed DBdatabase 
    # preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_mol2ind = preprocessed_DB[2]
    dict_prot2ind = preprocessed_DB[5]

    # # get the train dataset
    # train_set = get_list_couples_train(seed,
    #                                   preprocessed_DB)
    # true_inter = train_set[0]
    # false_inter = train_set[1]
    # true_inter = train_dataset[0]
    # false_inter = train_dataset[1]
    true_inter = train_dataset.true_inter
    false_inter = train_dataset.false_inter

    list_couples = true_inter.list_couples + false_inter.list_couples 
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

    # y_train = np.concatenate((true_inter.interaction_bool, 
    #                          false_inter.interaction_bool),
    #                          axis=0)

    # return K_train, y_train, true_inter, false_inter
    return K_train