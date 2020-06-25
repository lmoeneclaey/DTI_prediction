import argparse 
import pandas as pd
import pickle
import numpy as np
import os

from sklearn.svm import SVC
from datetime import datetime

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, check_protein, get_couples_from_array
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot
from DTI_prediction.predict.kronSVM.make_K_predict import make_K_predict_drug, make_K_predict_prot
from DTI_prediction.predict.predictions_postprocess import predictions_postprocess_drug

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

    parser.add_argument("--protein", default = False, action="strore_true",
                        help = "if you want to predict the drug targets of a protein")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels")

    args = parser.parse_args()

    # raw data directory
    raw_data_dir = 'data/' + args.DB_version + '/raw/'
    # pattern_name variable
    pattern_name = args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name + '/'

    #create directory
    if not os.path.exists(root + data_dir + '/' + 'predictions'):
        os.mkdir(root + data_dir + '/' + 'predictions')
        print("Predictions directory for", pattern_name, ",", args.DB_version,
        "created.")
    else:
        print("Predictions directory for", pattern_name, ",", args.DB_version,
        "already exists.")

    pred_dirname = root + data_dir + 'predictions/'
    clf_dirname = root + data_dir + 'classifiers/kronSVM/'

    now = datetime.now()
    date_time = now.strftime("%Y%m%d")

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    
    nb_mol = preprocessed_DB.drugs.nb 
    nb_prot = preprocessed_DB.proteins.nb

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.norm)

    # get the classifiers
    if args.norm == True:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_norm.data'
        output_filename = pred_dirname + pattern_name + '_kronSVM_norm_' + \
            args.dbid + '_pred_output_' + date_time + '.data'
        clean_filename = pred_dirname + pattern_name + "_kronSVM_norm_" + \
            args.dbid + '_pred_clean_' + date_time + '.csv'
    else:
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf.data'
        output_filename = pred_dirname + pattern_name + '_kronSVM_' + args.dbid\
        + '_pred_output_' + date_time + '.data'
        clean_filename = pred_dirname + pattern_name + '_kronSVM_' +args.dbid\
        + '_pred_clean_' + date_time + '.csv'
    list_clf = pickle.load(open(clf_filename, 'rb'))
    nb_clf = len(list_clf)

    # couples_filename = clf_dirname + pattern_name + \
    #     '_kronSVM_list_couples_of_clf.data'
    # list_couples_of_clf = pickle.load(open(couples_filename, 'rb'))
    train_datasets_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_train_datasets.data'
    list_train_datasets = pickle.load(open(train_datasets_filename, 'rb'))

    list_couples_of_clf = []
    for iclf in range(nb_clf):
        train_dataset = list_train_datasets[iclf]
        train_couples = get_couples_from_array(train_dataset)
        list_couples_of_clf.append(train_couples.list_couples)



    # You want to predict the targets of a drug
    if args.protein == False:
        
        if check_drug(args.dbid, preprocessed_DB.drugs)==True:

            pred = np.zeros((nb_prot, nb_clf))

            for clf_id in range(nb_clf):
                K_predict = make_K_predict_drug(args.dbid,
                                                preprocessed_DB,
                                                kernels,
                                                list_couples_of_clf[clf_id])

                pred[:, clf_id] = list_clf[clf_id].predict_proba(K_predict)[:,1]

        else:
            print("The drug you want to predict the targets is not in the database.")

            # new DB


    # You want to predict the liganfs of a protein
    else:

        if check_protein(args.dbid, preprocessed_DB.proteins)==False:
            print("The protein you want to predict the ligands is not in \
                    the database.")

        pred = np.zeros((nb_mol, nb_clf))

        for clf_id in range(nb_clf):
            K_predict = make_K_predict_prot(args.dbid,
                                            preprocessed_DB,
                                            kernels,
                                            list_couples_of_clf[clf_id])
            pred[:, clf_id] = list_clf[clf_id].predict_proba(K_predict)[:,1]
        
    print("Prediction for (classifier)", clf_id, "done.") 

    pickle.dump(pred, open(output_filename, 'wb'))

    # Post-processing

    raw_df = pd.read_csv(root + raw_data_dir + \
                         'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                         sep=",")
    raw_df = raw_df.fillna('')

    pred_clean = predictions_postprocess_drug(predictions_output=pred,
                                              DB=preprocessed_DB,
                                              raw_proteins_df=raw_df)

    pred_clean_final = pred_clean.drop_duplicates()
    pred_clean_final.to_csv(clean_filename)

    print("Predictions done and saved.")