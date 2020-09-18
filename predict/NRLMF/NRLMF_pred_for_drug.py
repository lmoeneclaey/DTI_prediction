import argparse
import copy 
import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, get_couples_from_array, add_drug, get_intMat
from DTI_prediction.process_dataset.get_molecules_smiles import get_non_DrugBank_smile
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot
from DTI_prediction.make_kernels.make_K_mol import make_mol_kernel, normalise_kernel, center_and_normalise_kernel

from DTI_prediction.make_classifiers.NRLMF_clf.NRLMF_utils import NRLMF

from DTI_prediction.predict.predictions_postprocess import predictions_postprocess_drug

root = '../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions with a NRLMF classifier."

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    parser.add_argument("dbid", type=str,
                        help = "the DrugBankId of the molecule/protein of which\
                        we want to predict the interactions")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels")

    parser.add_argument("--center_norm", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

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

    pred_dirname = root + data_dir + 'predictions/'
    train_datasets_dirname = root + data_dir + '/classifiers/train_datasets/'


    # Get the drugs and the proteins of the DrugBank database
    DB = get_DB(args.DB_version, args.DB_type)

    # Get the kernels of the DrugBank database
    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.center_norm, args.norm)
    DB_drugs_kernel = kernels[0]
    DB_proteins_kernel = kernels[1]

    # Modify the drugs kernel if the drug is not in the DrugBank database
    if check_drug(args.dbid, DB.drugs)==True:

        DB_drugs_final = copy.deepcopy(DB.drugs)    
        DB_drugs_kernel_final = copy.deepcopy(DB_drugs_kernel)
        intMat_final = (DB.intMat).T

    else:
        print(args.dbid, "is not in the database.")

        # read the corresponding sdf file
        drug_smile = get_non_DrugBank_smile(args.dbid)
        print("The sdf file for", args.dbid, "is downloaded.")

        # add the drug to the list of drugs
        DB_drugs_final = add_drug(drugs = DB.drugs,
                                  drug_id = args.dbid,
                                  smile = drug_smile)

        # make_new_K_mol, the normalise part should be included after confirmation
        DB_drugs_kernel_updated = make_mol_kernel(DB_drugs_final)[1]

        if args.center_norm == True:
            DB_drugs_kernel_final = center_and_normalise_kernel(DB_drugs_kernel_updated)
        elif args.norm == True:
            DB_drugs_kernel_final = normalise_kernel(DB_drugs_kernel_updated)

        intMat_updated = get_intMat(DB_drugs_final, DB.proteins, DB.couples)
        intMat_final = intMat_updated.T

    # Get the train datasets
    train_datasets_array_filename = train_datasets_dirname + pattern_name + \
        '_train_datasets_array.data'
    train_datasets_array = pickle.load(open(train_datasets_array_filename, 'rb'))

    # nb_clf = len(train_datasets_array)
    # print(nb_clf)

    # list_train_datasets = []
    # for iclf in range(nb_clf):
    #     train_dataset = get_couples_from_array(train_datasets_array[iclf])
    #     list_train_datasets.append(train_dataset)

    # pred = np.zeros((DB.proteins.nb, nb_clf))
    
    # for iclf in range(nb_clf):

    # train_dataset = list_train_datasets[iclf]

    # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
    # W = np.zeros(intMat.shape)
    # for prot_id, mol_id in train_dataset.list_couples:
    #     W[DB.drugs.dict_mol2ind[mol_id], DB.proteins.dict_prot2ind[prot_id]] = 1

    train_dataset = pd.DataFrame(train_datasets_array[0], columns=['UniProt ID', 
                                                                   'DrugbankID', 
                                                                   'interaction_bool'])
    train_true_interactions = train_dataset[train_dataset['interaction_bool']=='1']
    train_true_interactions_np = train_true_interactions.to_numpy()
    train_true = get_couples_from_array(train_true_interactions_np)

    train_intMat = np.zeros(intMat_final.shape)
    for prot_id, mol_id in train_true.list_couples:
        train_intMat[DB_drugs_final.dict_mol2ind[mol_id], DB.proteins.dict_prot2ind[prot_id]] = 1

    # Prepare the NRLMF classifier
    seed=92
    best_param = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, \
        'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, \
        'max_iter': 100}
    model = NRLMF(cfix=best_param['c'], 
                  K1=best_param['K1'], 
                  K2=best_param['K2'],
                  num_factors=best_param['r'],
                  lambda_d=best_param['lambda_d'],
                  lambda_t=best_param['lambda_t'], 
                  alpha=best_param['alpha'],
                  beta=best_param['beta'], 
                  theta=best_param['theta'],
                  max_iter=best_param['max_iter'])

    # model.fix_model(W=W,
    model.fix_model(W=train_intMat,
                    intMat=train_intMat, 
                    drugMat=DB_drugs_kernel_final, 
                    targetMat=DB_proteins_kernel, 
                    seed=seed)

    # Prepare the test dataset
    list_couples_predict = []
    for ind in range(DB.proteins.nb):
        list_couples_predict.append((DB_drugs_final.dict_mol2ind[args.dbid],ind)) 
        couples_predict_arr = np.array(list_couples_predict)

    # Process the predictions 
    predictions_output = model.predict(couples_predict_arr, train_intMat)

    pred_per_clf = []
    for mol_ind, prot_ind in couples_predict_arr:
        pred_per_clf.append(predictions_output[mol_ind, 
                                                prot_ind])

    predictions_df = pd.DataFrame({'UniProt ID' : list(DB.proteins.dict_protein.keys()),
                                   'prediction': pred_per_clf})
    
    # pred[:, iclf] = pred_per_clf

    # Post-processing ans saving file

    raw_df = pd.read_csv(root + raw_data_dir + \
                         'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                         sep=",")
    raw_df = raw_df.fillna('')

    # pred_clean = predictions_postprocess_drug(predictions_output=pred,
    #                                           DB=DB,
    #                                           raw_proteins_df=raw_df)

    # pred_clean_final = pred_clean.drop_duplicates()

    pred_clean_final = pd.merge(predictions_df,
                                raw_df[['UniProt ID', 'Gene Name', 'Name']],
                                left_on='UniProt ID',
                                right_on='UniProt ID')

    cols = ['UniProt ID', 'Gene Name', 'Name', 'prediction']
    pred_clean_final = pred_clean_final[cols]
    pred_clean_final = pred_clean_final.sort_values(by='prediction', 
                                                            ascending=False)
    pred_clean_final = pred_clean_final.drop_duplicates()

    # New DTI label

    dbid_interactions = train_dataset[(train_dataset['DrugbankID']==args.dbid) &
                                        (train_dataset['interaction_bool']=='1')]
    pred_clean_final['New DTI'] = ~pred_clean_final['UniProt ID'].isin(dbid_interactions['UniProt ID'])

    # add a column on new interactions

    now = datetime.now()
    date_time = now.strftime("%Y%m%d")
    if args.center_norm == True:
        clean_filename = pred_dirname + pattern_name + "_NRLMF_centered_norm_" + \
            args.dbid + '_pred_clean_' + date_time + '.csv'
    elif args.norm == True:
        clean_filename = pred_dirname + pattern_name + "_NRLMF_norm_" + \
            args.dbid + '_pred_clean_' + date_time + '.csv'
    else:
        clean_filename = pred_dirname + pattern_name + '_NRLMF_' +args.dbid\
        + '_pred_clean_' + date_time + '.csv'

    pred_clean_final.to_csv(clean_filename)

    print("Predictions done and saved.")

