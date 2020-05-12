import argparse
import numpy as np
import os
import pickle

import sklearn.model_selection as model_selection

from process_dataset.process_DB import get_DB
from make_K_train import ListInteractions, InteractionsTrainDataset

root = './../CFTR_PROJECT/'

def make_folds(seed, preprocessed_DB):
    """ 

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

    # Set the different seeds
    n_folds = 5
    true_seed = 62
    false_seed = 53

    # TRUE INTERACTIONS
    
    # get the interactions indices
    # ind_true_inter : indices where there is an interaction
    ind_true_inter = np.where(intMat == 1) 
    nb_true_inter = len(list_interactions)
    list_interactions_arr = np.array(list_interactions)

    # initialisation
    true_train_folds = []
    true_test_folds = []

    skf_true = model_selection.KFold(shuffle=True, random_state=true_seed)
    for train_index, test_index in skf_true.split(range(nb_true_inter)):
    
        train_fold = ListInteractions(list_couples=list_interactions_arr[train_index].tolist(),
                                      interaction_bool=np.array([1]*len(train_index)),
                                      ind_inter=(ind_true_inter[0][train_index],
                                                 ind_true_inter[1][train_index]))
    
        test_fold = ListInteractions(list_couples=list_interactions_arr[test_index].tolist(),
                                     interaction_bool=np.array([1]*len(test_index)),
                                     ind_inter=(ind_true_inter[0][test_index],
                                                ind_true_inter[1][test_index]))
    
        true_train_folds.append(train_fold)
        true_test_folds.append(test_fold)

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
    # in the train data set
    ind_false_inter = (ind_all_false_inter[0][mask], 
                       ind_all_false_inter[1][mask])
    nb_false_inter = len(ind_false_inter[0])

    list_false_inter = []
    for i in range(nb_false_inter):
        list_false_inter.append((dict_ind2prot[ind_false_inter[0][i]],
                                 dict_ind2mol[ind_false_inter[1][i]]))
    list_false_inter_arr = np.array(list_false_inter)

    # initialisation
    false_train_folds = []
    false_test_folds = []

    skf_false = model_selection.KFold(n_folds, shuffle=True, random_state=false_seed)
    for train_index, test_index in skf_false.split(range(nb_false_inter)):
    
        train_fold = ListInteractions(list_couples=list_false_inter_arr[train_index].tolist(),
                                    interaction_bool=np.array([0]*len(train_index)),
                                    ind_inter=(ind_false_inter[0][train_index],
                                                ind_false_inter[1][train_index]))
    
        test_fold = ListInteractions(list_couples=list_false_inter_arr[test_index].tolist(),
                                    interaction_bool=np.array([0]*len(test_index)),
                                    ind_inter=(ind_false_inter[0][test_index],
                                                ind_false_inter[1][test_index]))
    
        false_train_folds.append(train_fold)
        false_test_folds.append(test_fold)

    print("List of the couples for all the folds done.")

    # CONCATENATION
    # Concatenate "true" and "false" interacitons to create the final train and 
    # test folds

    train_folds = []
    test_folds = []
    for ifold in range(n_folds):
        
        train_ifold = InteractionsTrainDataset(true_inter=true_train_folds[ifold],
                                               false_inter=false_train_folds[ifold])    
        train_folds.append(train_ifold)

        test_folds.append(true_test_folds[ifold] + false_test_folds[ifold])

    print("Train folds and test folds prepared.")

    return train_folds, test_folds

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Create train and test folds for a particular dataset and save it.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name = args.process_name + '_' + args.DB_type
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + pattern_name + '/'

    #create directories
    if not os.path.exists(root + data_dir + '/' + 'CrossValidation'):
        os.mkdir(root + data_dir + '/' + 'CrossValidation')
        print("Cross Validation directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("Cross Validation directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'CrossValidation/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type, args.process_name)

    seed = 242
    train_folds, test_folds = make_folds(seed, preprocessed_DB)

    train_folds_filename = cv_dirname + pattern_name + \
        '_train_folds.data'
    test_folds_filename = cv_dirname + pattern_name + \
        '_test_folds.data'    

    pickle.dump(train_folds,
                open(train_folds_filename, 'wb'))
    pickle.dump(test_folds,
                open(test_folds_filename, 'wb'))





