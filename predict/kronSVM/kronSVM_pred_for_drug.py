import argparse 
import pandas as pd
import pickle
import numpy as np
import os

from sklearn.svm import SVC
from datetime import datetime

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, get_couples_from_array, add_drug
from DTI_prediction.process_dataset.get_molecules_smiles import get_non_DrugBank_smile
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot
from DTI_prediction.make_kernels.make_K_mol import make_mol_kernel, normalise_kernel, center_and_normalise_kernel
from DTI_prediction.predict.kronSVM.make_K_predict_for_drug import make_K_predict_drug
from DTI_prediction.predict.predictions_postprocess import predictions_postprocess_drug

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions with a kronSVM classifier."

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

    parser.add_argument("--non_balanced", default = False, action="store_true",
                        help = "whether the train dataset are balanced in terms \
                            of nb of interactions per proteins and drugs, \
                            balanced by default")

    parser.add_argument("--output_path", type=str,
                        help = "if the predictions need to be in another folder.")

    # parser.add_argument("--norm", default = False, action="store_true", 
    #                     help = "whether or not to normalize the kernels")

    # parser.add_argument("--center_norm", default = False, action="store_true", 
    #                     help = "whether or not to center AND normalize the \
    #                         kernels, False by default")

    args = parser.parse_args()

    # raw data directory
    raw_data_dir = 'data/' + args.DB_version + '/raw/'
    # pattern_name variable
    pattern_name = args.DB_type + '_' + args.process_name
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/' + pattern_name + '/'


    if args.output_path is None:

        #create directory
        if not os.path.exists(root + data_dir + '/' + 'predictions'):
            os.mkdir(root + data_dir + '/' + 'predictions')
            print("Predictions directory for", pattern_name, ",", args.DB_version,
            "created.")
        if not os.path.exists(root + data_dir + '/' + 'predictions/kronSVM'):
            os.mkdir(root + data_dir + '/' + 'predictions/kronSVM')
            print("kronSVM predictions directory for", pattern_name, ",", args.DB_version,
            "created.")

        pred_dirname = root + data_dir + 'predictions/kronSVM/'

    else:

        if os.path.exists(args.output_path):
            pred_dirname = args.output_path
        else:
            print("ERROR: The informed output path is doesn't exist.")
    
    clf_dirname = root + data_dir + 'classifiers/kronSVM/'
    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'

    now = datetime.now()
    date_time = now.strftime("%Y%m%d")

    # if args.center_norm == True:
    #     clf_filename = clf_dirname + pattern_name + \
    #     '_kronSVM_list_clf_centered_norm.data'
    #     output_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + \
    #         args.dbid + '_pred_output_' + date_time + '.data'
    #     clean_filename = pred_dirname + pattern_name + "_kronSVM_centered_norm_" + \
    #         args.dbid + '_pred_clean_' + date_time + '.csv'
    # elif args.norm == True:
    #     clf_filename = clf_dirname + pattern_name + \
    #     '_kronSVM_list_clf_norm.data'
    #     output_filename = pred_dirname + pattern_name + '_kronSVM_norm_' + \
    #         args.dbid + '_pred_output_' + date_time + '.data'
    #     clean_filename = pred_dirname + pattern_name + "_kronSVM_norm_" + \
    #         args.dbid + '_pred_clean_' + date_time + '.csv'
    # else:
    #     clf_filename = clf_dirname + pattern_name + \
    #     '_kronSVM_list_clf.data'
    #     output_filename = pred_dirname + pattern_name + '_kronSVM_' + args.dbid\
    #     + '_pred_output_' + date_time + '.data'
    #     clean_filename = pred_dirname + pattern_name + '_kronSVM_' +args.dbid\
    #     + '_pred_clean_' + date_time + '.csv'

    # output_filename = pred_dirname + pattern_name + '_kronSVM_centered_norm_' + \
    #     args.dbid + '_pred_output_' + date_time + '.data'


    # Get the drugs and the proteins of the DrugBank database
    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    DB_drugs = preprocessed_DB.drugs
    DB_proteins = preprocessed_DB.proteins

    # Get the kernels of the DrugBank database
    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type)
    DB_drugs_kernel = kernels[0]
    DB_proteins_kernel = kernels[1]
    
    if args.non_balanced == True:

        train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_non_balanced_train_datasets_array.data'
        clf_filename = clf_dirname + pattern_name + \
        '_non_balanced_kronSVM_list_clf_centered_norm.data'
        clean_filename = pred_dirname + pattern_name + "_non_balanced_kronSVM_centered_norm_" + \
        args.dbid + '_pred_clean_' + date_time + '.csv'

    else:

        train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'
        clf_filename = clf_dirname + pattern_name + \
        '_kronSVM_list_clf_centered_norm.data'
        clean_filename = pred_dirname + pattern_name + "_kronSVM_centered_norm_" + \
        args.dbid + '_pred_clean_' + date_time + '.csv'

    # get the classifiers
    list_clf = pickle.load(open(clf_filename, 'rb'))
    nb_clf = len(list_clf)

    # Get the train datasets
    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    nb_clf = len(train_datasets_array)

    list_train_datasets = []
    for iclf in range(nb_clf):
        train_dataset = get_couples_from_array(train_datasets_array[iclf])
        list_train_datasets.append(train_dataset)

    # Process the predictions
    pred = np.zeros((DB_proteins.nb, nb_clf))

    # If the drug is in the DrugBank database
    if check_drug(args.dbid, preprocessed_DB.drugs)==True:

        for iclf in range(nb_clf):

            train_dataset = list_train_datasets[iclf]

            K_predict = make_K_predict_drug(dbid = args.dbid,
                                            drugs = preprocessed_DB.drugs,
                                            proteins = preprocessed_DB.proteins,
                                            drugs_kernel = DB_drugs_kernel,
                                            proteins_kernel = DB_proteins_kernel,
                                            list_couples_train = train_dataset.list_couples)

            pred[:, iclf] = list_clf[iclf].predict_proba(K_predict)[:,1]

    else:
        print(args.dbid, "is not in the database.")

        # read the corresponding sdf file
        drug_smile = get_non_DrugBank_smile(args.dbid)
        print("The sdf file for", args.dbid, "is downloaded.")
        # add the drug to the list of drugs
        DB_drugs_updated = add_drug(drugs = preprocessed_DB.drugs,
                                    drug_id = args.dbid,
                                    smile = drug_smile)
        # make_new_K_mol, the normalise part should be included after confirmation
        DB_drugs_kernel_updated = make_mol_kernel(DB_drugs_updated)[1]

        # if args.center_norm == True:
        #     DB_drugs_kernel_updated = center_and_normalise_kernel(DB_drugs_kernel_updated)
        # elif args.norm == True:
        #     DB_drugs_kernel_updated = normalise_kernel(DB_drugs_kernel_updated)

        DB_drugs_kernel_updated = center_and_normalise_kernel(DB_drugs_kernel_updated)

        for iclf in range(nb_clf):

            train_dataset = list_train_datasets[iclf]

            K_predict = make_K_predict_drug(dbid = args.dbid,
                                            drugs = DB_drugs_updated,
                                            proteins = preprocessed_DB.proteins,
                                            drugs_kernel = DB_drugs_kernel_updated,
                                            proteins_kernel = DB_proteins_kernel,
                                            list_couples_train = train_dataset.list_couples)

            pred[:, iclf] = list_clf[iclf].predict_proba(K_predict)[:,1]

    # Post-processing

    raw_df = pd.read_csv(root + raw_data_dir + \
                         'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                         sep=",")
    raw_df = raw_df.fillna('')

    pred_clean = predictions_postprocess_drug(predictions_output=pred,
                                              DB=preprocessed_DB,
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