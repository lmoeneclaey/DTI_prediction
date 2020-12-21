import argparse
import numpy as np
import os
import pickle

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.NRLMF_clf.NRLMF_utils import NRLMF
from DTI_prediction.process_dataset.DB_utils import get_couples_from_array
root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Cross validation analysis of the NRLMF algorithm.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    # add the arguments here if you want to check the arguments

    # parser.add_argument("--norm", default = False, action="store_true", 
    #                     help = "where or not to normalize the kernels, False \
    #                     by default")

    # parser.add_argument("--center_norm", default = False, action="store_true", 
    #                     help = "whether or not to center AND normalize the \
    #                         kernels, False by default")

    parser.add_argument("--balanced_on_proteins", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--balanced_on_drugs", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    if not os.path.exists(root + data_dir + '/' + 'cross_validation/NRLMF'):
        os.mkdir(root + data_dir + '/' + 'cross_validation/NRLMF')
        print("NRLMF cross validation directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("NRLMF cross validation directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    cv_dirname = root + data_dir + 'cross_validation/'
    nrlmf_cv_dirname = root + data_dir + 'cross_validation/NRLMF/'

    DB = get_DB(args.DB_version, args.DB_type)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, center_norm=True, norm=False)
    DB_drugs_kernel = kernels[0]
    DB_proteins_kernel = kernels[1]

    # Get the nested folds
    nested_cv_dirname = root + data_dir + 'cross_validation/nested_folds/'

    if args.balanced_on_proteins == True:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_double_balanced_1_clf_array.data'
            output_filename = nrlmf_cv_dirname + args.DB_type + \
            '_NRLMF_cv_nested_double_balanced_1_clf_pred.data'
        else:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_balanced_on_proteins_1_clf_array.data'
            output_filename = nrlmf_cv_dirname + args.DB_type + \
            '_NRLMF_cv_nested_balanced_on_proteins_1_clf_pred.data'
    else:
        if args.balanced_on_drugs == True:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_balanced_on_drugs_1_clf_array.data'
            output_filename = nrlmf_cv_dirname + args.DB_type + \
            '_NRLMF_cv_nested_balanced_on_drugs_1_clf_pred.data'
        else:
            nested_folds_array_filename = nested_cv_dirname \
            + args.DB_type + '_nested_folds_non_balanced_1_clf_array.data'
            output_filename = nrlmf_cv_dirname + args.DB_type + \
            '_NRLMF_cv_nested_non_balanced_1_clf_pred.data'

    nested_folds_array = pickle.load(open(nested_folds_array_filename, 'rb'))

    list_folds = []
    for ifold in range(len(nested_folds_array)):
        fold_dataset = get_couples_from_array(nested_folds_array[ifold])
        list_folds.append(fold_dataset)

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
    intMat = (DB.intMat).T

    nb_folds = len(list_folds)

    cv_pred = []

    for ifold in range(nb_folds):

        print("fold:", ifold)

        test_couples = list_folds[ifold].list_couples
        list_couples_predict = []
        for prot_id, mol_id in test_couples:
            list_couples_predict.append((DB.drugs.dict_mol2ind[mol_id], 
                                         DB.proteins.dict_prot2ind[prot_id]))
        couples_predict_arr = np.array(list_couples_predict)

        train_folds = [list_folds[(ifold+1)%5],
                         list_folds[(ifold+2)%5],
                         list_folds[(ifold+4)%5],
                         list_folds[(ifold+5)%5]]

        train_dataset = sum(train_folds)

        # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
        W = np.zeros(intMat.shape)
        for prot_id, mol_id in train_dataset.list_couples:
            W[DB.drugs.dict_mol2ind[mol_id], DB.proteins.dict_prot2ind[prot_id]] = 1

        # R is a filter of W on intMat
        R = W * intMat

        model.fix_model(W=W,
                        intMat=intMat, 
                        drugMat=DB_drugs_kernel, 
                        targetMat=DB_proteins_kernel, 
                        seed=seed)

        # Process the predictions 
        predictions_output = model.predict(test_data=couples_predict_arr, 
                                            intMat_for_verbose=intMat)

        pred = []
        for mol_ind, prot_ind in couples_predict_arr:
            pred.append(predictions_output[mol_ind, prot_ind])

        cv_pred.append(np.array(pred))
        print("Prediction for fold", ifold, "done.") 

    pickle.dump(cv_pred, open(output_filename, 'wb'))
    print("Cross validation done and saved.")