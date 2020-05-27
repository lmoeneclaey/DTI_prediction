import argparse 
import pickle
import numpy as np
import os

from sklearn.svm import SVC

from process_dataset.process_DB import get_DB
from make_K_inter import get_K_mol_K_prot
from make_K_predict import make_K_predict

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions "

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    parser.add_argument("dbid", type=str,
                        help = "the DrugBankId of the molecule/protein of which\
                        we want to predict the interactions")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "where or not to normalize the kernels")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name = args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name + '/'

    #create directory
    if not os.path.exists(root + data_dir + '/' + 'Predictions'):
        os.mkdir(root + data_dir + '/' + 'Predictions')
        print("Predictions directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("Predictions directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    pred_dirname = root + data_dir + 'Predictions/'
    clf_dirname = root + data_dir + 'Classifiers/kronSVM/'

    preprocessed_DB = get_DB(args.DB_version, args.DB_type, args.process_name)
    dict_ind2mol = preprocessed_DB[1]
    nb_mol = len(dict_ind2mol)
    dict_ind2prot = preprocessed_DB[4]
    nb_prot = len(dict_ind2prot)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.process_name,
                               args.norm)

    # add forbidden_list

    # get the classifiers
    if args.norm == True:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_norm.data'
        output_filename = pred_dirname + pattern_name + '_kronSVM_norm_' + \
            args.dbid + '_pred.data'
    else:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf.data'
        output_filename = pred_dirname + pattern_name + '_kronSVM_' + args.dbid\
        + '_pred.data'
    list_clf = pickle.load(open(clf_filename, 'rb'))
    nb_clf = len(list_clf)

    couples_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_couples_of_clf.data'
    list_couples_of_clf = pickle.load(open(couples_filename, 'rb'))

    # initialisation
    if args.dbid[:2] == 'DB':
        # The molecule is a drug
        pred = np.zeros((nb_prot, nb_clf))
    else:
        # The molecule is a protein
        pred = np.zeros((nb_mol, nb_clf))

    for clf_id in range(nb_clf):
        K_predict = make_K_predict(args.dbid,
                                   preprocessed_DB,
                                   kernels,
                                   list_couples_of_clf[clf_id])
        pred[:, clf_id] = list_clf[clf_id].predict_proba(K_predict)[:,1]
        print("Prediction for (classifier)", clf_id, "done.") 

    pickle.dump(pred, open(output_filename, 'wb'))
    print("Predictions done and saved.")