import numpy as np

from DTI_prediction.process_dataset.DB_utils import Couples

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
        List of all the "true" interactions in a 'Couples' class and \
        all the "false" interactions, also in a 'Couples' class. 
    true_inter : Couples
        List of all the "true" interactions in the train dataset
    false_inter : Couples
        List of all the "false" interactions in the train dataset
    """ 

    dict_ind2mol = preprocessed_DB.drugs.dict_ind2mol
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot
    intMat = preprocessed_DB.intMat
    interactions = preprocessed_DB.interactions

    # TRUE INTERACTIONS

    true_inter = interactions

    # "FALSE" INTERACTIONS

    # get the interactions indices
    # ind_false_inter : indices where there is not an interaction
    ind_all_false_inter = np.where(intMat == 0)
    nb_all_false_inter = len(ind_all_false_inter[0])

    # choose between all the "false" interactions, nb_true_interactions couples,
    # without replacement
    np.random.seed(seed)
    mask = np.random.choice(np.arange(nb_all_false_inter), 
                            interactions.nb,
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

    false_inter = Couples(list_couples=list_false_inter,
                          interaction_bool=np.array([0]*nb_false_inter).reshape(-1,1))

    print("list of all the couples done.")

    # train_dataset = [true_inter, false_inter]
    train_dataset = InteractionsTrainDataset(true_inter=true_inter,
                                             false_inter=false_inter)
    print("Train dataset done.")

    # return true_inter, false_inter
    return train_dataset

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
    dict_mol2ind = preprocessed_DB.drugs.dict_mol2ind
    dict_prot2ind = preprocessed_DB.proteins.dict_prot2ind

    # # get the train dataset
    true_inter = train_dataset.true_inter
    false_inter = train_dataset.false_inter

    list_couples = true_inter.list_couples + false_inter.list_couples 
    nb_couples = len(list_couples)

    # get the kernels
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