import argparse 
import pandas as pd
import pickle
import numpy as np
import os

from datetime import datetime

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, get_couples_from_array, add_drug
from DTI_prediction.process_dataset.get_molecules_smiles import get_non_DrugBank_smile, get_non_DrugBank_feature
from DTI_prediction.process_dataset.process_DB import get_DB

from DTI_prediction.make_kernels.get_kernels import get_mol_prot_features

from DTI_prediction.make_classifiers.RF_clf.make_RF_clf import get_Xcouple
from DTI_prediction.predict.predictions_postprocess import predictions_postprocess_drug

from sklearn.ensemble import RandomForestClassifier


root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions with a Ranfom Forest classifier."

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
    if not os.path.exists(root + data_dir + '/' + 'predictions/RF'):
        os.mkdir(root + data_dir + '/' + 'predictions/RF')
        print("Random Forest predictions directory for", pattern_name, ",", args.DB_version,
        "created.")

    pred_dirname = root + data_dir + 'predictions/RF/'
    clf_dirname = root + data_dir + '/classifiers/RF/'
    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'

    now = datetime.now()
    date_time = now.strftime("%Y%m%d")

    clf_filename = clf_dirname + pattern_name + '_RF_list_clf.data'
    clean_filename = pred_dirname + pattern_name + '_RF_' +args.dbid\
    + '_pred_clean_' + date_time + '.csv'

    # Get the drugs and the proteins of the DrugBank database
    DB = get_DB(args.DB_version, args.DB_type)
    DB_drugs = DB.drugs
    DB_proteins = DB.proteins

    # Get the features
    features = get_mol_prot_features(args.DB_version, args.DB_type)
    X_mol = features[0]
    X_prot = features[1]
    
    # get the classifiers
    list_clf = pickle.load(open(clf_filename, 'rb'))
    nb_clf = len(list_clf)

    # Get the train datasets
    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'
    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    nb_clf = len(train_datasets_array)

    list_train_datasets = []
    for iclf in range(nb_clf):
        train_dataset = get_couples_from_array(train_datasets_array[iclf])
        list_train_datasets.append(train_dataset)

    # Process the predictions
    pred = np.zeros((DB_proteins.nb, nb_clf))

    # If the drug is in the DrugBank database
    if check_drug(args.dbid, DB.drugs)==True:

        list_couples_predict = []
        for ind in range(DB.proteins.nb):
            list_couples_predict.append((DB.proteins.dict_ind2prot[ind],args.dbid))

        X_pred = get_Xcouple(list_couples_predict,
                             X_mol,
                             X_prot,
                             DB.drugs.dict_mol2ind,
                             DB.proteins.dict_prot2ind)

        for iclf in range(nb_clf):
    
            pred[:, iclf] = list_clf[iclf].predict_proba(X_pred)[:,1]

    else:
        print(args.dbid, "is not in the database.")

        # read the corresponding sdf file
        drug_smile = get_non_DrugBank_smile(args.dbid)
        print("The sdf file for", args.dbid, "is downloaded.")

        drug_feature = get_non_DrugBank_feature(drug_smile).reshape(1,-1)

        X_pred = np.zeros((DB.proteins.nb, drug_feature.shape[1]+X_prot.shape[1]))
        for i in range(DB.proteins.nb):
            X_pred[i,:] = np.concatenate([drug_feature[0,:], X_prot[i,:]])

        for iclf in range(nb_clf):

            pred[:, iclf] = list_clf[iclf].predict_proba(X_pred)[:,1]

    # Post-processing

    raw_df = pd.read_csv(root + raw_data_dir + \
                         'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                         sep=",")
    raw_df = raw_df.fillna('')

    pred_clean = predictions_postprocess_drug(predictions_output=pred,
                                              DB=DB,
                                              raw_proteins_df=raw_df)

    pred_clean_final = pred_clean.drop_duplicates()

    # New DTI label
    train_dataset = pd.DataFrame(train_datasets_array[0], columns=['UniProt ID', 
                                                                   'DrugbankID', 
                                                                   'interaction_bool'])

    dbid_interactions = train_dataset[(train_dataset['DrugbankID']==args.dbid) &
                                        (train_dataset['interaction_bool']=='1')]
    pred_clean_final['New DTI'] = ~pred_clean_final['UniProt ID'].isin(dbid_interactions['UniProt ID'])

    pred_clean_final.to_csv(clean_filename)
    print("Predictions done and saved.")