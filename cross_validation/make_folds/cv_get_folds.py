import numpy as np
import pickle

from process_dataset.DB_utils import ListInteractions
from make_K_train import InteractionsTrainDataset

root = "./../CFTR_PROJECT/"

def get_test_folds(DB_version, DB_type, process_name):

    """ 
    Load the test folds

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'

    Returns
    -------
    test_folds : ListInteractions 
    """ 

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    cv_dirname = root + data_dir + 'CrossValidation/'

    test_folds_list_couples_filename = cv_dirname + 'test_folds/' \
        + pattern_name + '_test_folds_list_couples.data'
    test_folds_interaction_bool_filename = cv_dirname + 'test_folds/' \
        + pattern_name + '_test_folds_interaction_bool.data'
    test_folds_ind_inter_filename = cv_dirname + 'test_folds/' \
        + pattern_name + '_test_folds_ind_inter.data'

    test_folds_list_couples = pickle.load(open(test_folds_list_couples_filename, 'rb'))
    test_folds_interaction_bool = pickle.load(open(test_folds_interaction_bool_filename, 'rb'))
    test_folds_ind_inter = pickle.load(open(test_folds_ind_inter_filename, 'rb'))

    nb_folds = len(test_folds_list_couples)
    test_folds = []
    
    for ifold in range(nb_folds):

        test_fold = ListInteractions(list_couples=test_folds_list_couples[ifold],
                                     interaction_bool=test_folds_interaction_bool[ifold],
                                     ind_inter=test_folds_ind_inter[ifold])

        test_folds.append(test_fold)

    return test_folds

def get_train_folds(DB_version, DB_type, process_name):

    """ 
    Load the train folds

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'

    Returns
    -------
    train_folds : InteractionsTrainDataset
    """ 

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    cv_dirname = root + data_dir + 'CrossValidation/'

    # True interactions

    train_true_folds_list_couples_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_true_folds_list_couples.data'
    train_true_folds_interaction_bool_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_true_folds_interaction_bool.data'
    train_true_folds_ind_inter_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_true_folds_ind_inter.data'

    train_true_folds_list_couples = pickle.load(open(train_true_folds_list_couples_filename, 'rb'))
    train_true_folds_interaction_bool = pickle.load(open(train_true_folds_interaction_bool_filename, 'rb'))
    train_true_folds_ind_inter = pickle.load(open(train_true_folds_ind_inter_filename, 'rb'))

    nb_folds = len(train_true_folds_list_couples)
    train_true_folds = []
    
    for ifold in range(nb_folds):

        train_true_fold = ListInteractions(list_couples=train_true_folds_list_couples[ifold],
                                     interaction_bool=train_true_folds_interaction_bool[ifold],
                                     ind_inter=train_true_folds_ind_inter[ifold])

        train_true_folds.append(train_true_fold)

    # False interactions

    train_false_folds_list_couples_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_false_folds_list_couples.data'
    train_false_folds_interaction_bool_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_false_folds_interaction_bool.data'
    train_false_folds_ind_inter_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_false_folds_ind_inter.data'

    train_false_folds_list_couples = pickle.load(open(train_false_folds_list_couples_filename, 'rb'))
    train_false_folds_interaction_bool = pickle.load(open(train_false_folds_interaction_bool_filename, 'rb'))
    train_false_folds_ind_inter = pickle.load(open(train_false_folds_ind_inter_filename, 'rb'))

    train_false_folds = []
    for ifold in range(nb_folds):

        train_false_fold = ListInteractions(list_couples=train_false_folds_list_couples[ifold],
                                     interaction_bool=train_false_folds_interaction_bool[ifold],
                                     ind_inter=train_false_folds_ind_inter[ifold])

        train_false_folds.append(train_false_fold)

    train_folds = []
    for ifold in range(nb_folds):
        train_ifold = InteractionsTrainDataset(true_inter=train_true_folds[ifold],
                                               false_inter=train_false_folds[ifold])    
        train_folds.append(train_ifold)

    return train_folds