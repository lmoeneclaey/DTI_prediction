import argparse
import numpy as np
import os
import pickle

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB 
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

from DTI_prediction.make_classifiers.NRLMF_clf.NRLMF_utils import NRLMF

from DTI_prediction.cross_validation.make_folds.cv_get_folds import get_train_folds, get_test_folds

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

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "where or not to normalize the kernels, False \
                        by default")

    parser.add_argument("--center_norm", default = False, action="store_true", 
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

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.center_norm, args.norm)
    DB_drugs_kernel = kernels[0]
    DB_proteins_kernel = kernels[1]

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

    # Get the train datasets
    train_folds = get_train_folds(args.DB_version, args.DB_type)
    # Get the test datasets
    test_folds = get_test_folds(args.DB_version, args.DB_type)

    nb_folds = len(train_folds)
    nb_clf = len(train_folds[0])

    cv_pred = []

    for ifold in range(nb_folds):
        print("fold:", ifold)

        test_couples = test_folds[ifold].list_couples
        list_couples_predict = []
        for prot_id, mol_id in test_couples:
            list_couples_predict.append((DB.drugs.dict_mol2ind[mol_id], 
                                         DB.proteins.dict_prot2ind[prot_id]))
        couples_predict_arr = np.array(list_couples_predict)

        pred = []

        for iclf in range(nb_clf):

            train_dataset = train_folds[ifold][iclf]

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

            pred_per_clf = []
            for mol_ind, prot_ind in couples_predict_arr:
                pred_per_clf.append(predictions_output[mol_ind, 
                                                       prot_ind])

            pred.append(pred_per_clf)

        cv_pred.append(pred)
        print("Prediction for fold", ifold, "done.") 

    # get the classifiers
    if args.center_norm == True:
        output_filename = nrlmf_cv_dirname + args.DB_type + \
        '_NRLMF_cv_pred_centered_norm.data'
    elif args.norm == True:
        output_filename = nrlmf_cv_dirname + args.DB_type + \
        '_NRLMF_cv_pred_norm.data'
    else:
        output_filename = nrlmf_cv_dirname + args.DB_type + \
        '_NRLMF_cv_pred.data'

    pickle.dump(cv_pred, open(output_filename, 'wb'))
    print("Cross validation done and saved.")