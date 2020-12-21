import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_mol_prot_features

from sklearn.ensemble import RandomForestClassifier

root = './../CFTR_PROJECT/'

def get_Xcouple(x, X_mol, X_prot, dict_mol2ind, dict_prot2ind):
    X = np.zeros((len(x), X_mol.shape[1] + X_prot.shape[1]))
    for i in range(len(x)):
        prot, mol = x[i]
        X[i, :] = np.concatenate([X_mol[dict_mol2ind[mol], :],
                                  X_prot[dict_prot2ind[prot], :]])
    return X

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Process the kernel of interactions of a list of molecules and proteins \
        and create the corresponding Random Forest classifier.")

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
    pattern_name =  args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name

    #create directories
    if not os.path.exists(root + 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name):
        os.mkdir(root + 'data/' + args.DB_version + '/' + args.DB_type + '/' +  pattern_name)
        print("Directory", pattern_name, "for",  args.DB_version, "created")
    else: 
        print("Directory", pattern_name, "for",  args.DB_version, " already exists")

    if not os.path.exists(root + data_dir + '/' + 'classifiers'):
        os.mkdir(root + data_dir + '/' + 'classifiers')
        print("Classifiers directory for", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("Classifiers directory for", pattern_name, ",", args.DB_version,
        "already exists.")

    if not os.path.exists(root + data_dir + '/' + 'classifiers/RF'):
        os.mkdir(root + data_dir + '/' + 'classifiers/RF')
        print("RF classifiers directory for ", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("RF classifiers directory for ", pattern_name, ",", args.DB_version,
        "already exists.")

    # Get the train datasets 

    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'
    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'

    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    nb_clf = len(train_datasets_array)

    list_train_datasets = []
    for iclf in range(nb_clf):
        train_dataset = get_couples_from_array(train_datasets_array[iclf])
        list_train_datasets.append(train_dataset)

    clf_dirname = root + data_dir + '/classifiers/RF/'

    DB = get_DB(args.DB_version, args.DB_type)

    features = get_mol_prot_features(args.DB_version, args.DB_type)
    X_mol = features[0]
    X_prot = features[1]

    list_clf = []
    list_seed = [53, 223, 481, 21, 49]

    for iclf in range(nb_clf):

        ## train matrice
        train_couples = list_train_datasets[iclf]     
        X_tr = get_Xcouple(train_couples.list_couples,
                           X_mol,
                           X_prot,
                           DB.drugs.dict_mol2ind,
                           DB.proteins.dict_prot2ind)       
        y_tr = train_couples.interaction_bool.reshape(-1,)

        rf = RandomForestClassifier(n_estimators=600,
                                    min_samples_leaf=1,
                                    min_samples_split=5,
                                    max_depth=20, 
                                    random_state=list_seed[iclf])
        rf.fit(X_tr, y_tr)

        list_clf.append(rf)
    
    print("Classifiers done.")
        
    clf_filename = clf_dirname + pattern_name + '_RF_list_clf.data'

    pickle.dump(list_clf, 
                open(clf_filename, 'wb'),
                protocol=2)
    
    print("Classifiers saved.")