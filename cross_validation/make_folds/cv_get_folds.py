import numpy as np
import pickle

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB, get_couples_from_array
from DTI_prediction.make_classifiers.kronSVM_clf.make_K_train import InteractionsTrainDataset

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
    pattern_name =  DB_type + '_' + process_name
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name + '/'

    cv_dirname = root + data_dir + '/cross_validation/'

    test_folds_array_filename = cv_dirname + 'test_folds/' \
        + pattern_name + '_test_folds_array.data'

    test_folds_array = pickle.load(open(test_folds_array_filename, 'rb'))
    nb_folds = len(test_folds_array)

    test_folds = []
    for ifold in range(nb_folds):

        test_fold = get_couples_from_array(test_folds_array[ifold])

        test_folds.append(test_fold)

    return test_folds

def get_train_folds(DB_version, DB_type, process_name, nb_clf):

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
    pattern_name =  DB_type + '_' + process_name
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/' + pattern_name 

    cv_dirname = root + data_dir + '/cross_validation/'

    # True interactions

    train_true_folds_array_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_true_folds_array.data'

    train_true_folds_array = pickle.load(open(train_true_folds_array_filename, 'rb'))

    nb_folds = len(train_true_folds_array)
    train_true_folds = []
    
    for ifold in range(nb_folds):

        train_true_fold = get_couples_from_array(train_true_folds_array[ifold])
        train_true_folds.append(train_true_fold)

    # False interactions

    train_false_folds_array_filename = cv_dirname + 'train_folds/' \
        + pattern_name + '_train_false_folds_array.data'

    train_false_folds_array = pickle.load(open(train_false_folds_array_filename, 'rb'))

    train_datasets = []
    for ifold in range(nb_folds):
    
        train_datasets_per_fold = []
        for iclf in range(nb_clf):
        
            train_false_fold = get_couples_from_array(train_false_folds_array[iclf])
            train_dataset = InteractionsTrainDataset(true_inter=train_true_folds[ifold],
                                                     false_inter=train_false_fold)

            train_datasets_per_fold.append(train_dataset)
        
        train_datasets.append(train_datasets_per_fold)

    return train_datasets