
if __name__ == "__main__":
    # get command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--dataname', type=str, help='name of dataset')
    parser.add_argument('-nf', '--n_folds', type=int, help='nb of folds')
    parser.add_argument('-tef', '--test_fold', type=int, help='which fold to test on')
    parser.add_argument('-valf', '--val_fold', type=int,
                        help='which fold to use a validation')
    parser.add_argument('-set', '--setting', type=int,
                        help='setting of CV either 1,2,3,4')
    parser.add_argument('-rtr', '--ratio_tr', type=int,
                        help='ratio pos/neg in train either 1,2,5')
    parser.add_argument('-rte', '--ratio_te', type=int,
                        help='ratio pos/neg in test either 1,2,5')
    parser.add_argument('-cvv', '--cv_val', action='store_true', default=False,
                        help='true if in command line, else false')
    args = parser.parse_args()
    DB, fold_val, fold_te, setting, ratio_tr, ratio_te = \
        args.dataname, args.val_fold, args.test_fold, args.setting, args.ratio_tr, args.ratio_te

    # get dataset specific kernels and items
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)
    drugMat = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
    targetMat = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
    n_folds, seed = 5, 92

    # Récupération des folds de cross validation

    # get folds of data
    # x_tr, x_val, x_te are list of (protein, molecule) pairs (of IDs)
    #       for training, validation, test
    # y_tr, y_val, y_te are the true labels associated with the pairs
    if setting != 4:
        x_tr = [ind for sub in [pickle.load(open(data_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(n_folds) if i != fold_val and i != fold_te]
                for ind in sub]
        y_tr = [ind for sub in [pickle.load(open(y_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(n_folds) if i != fold_val and i != fold_te]
                for ind in sub]
        x_val = pickle.load(open(data_file(DB, fold_val, setting, ratio_te), 'rb'))
        y_val = pickle.load(open(y_file(DB, fold_val, setting, ratio_te), 'rb'))
        x_te = pickle.load(open(data_file(DB, fold_te, setting, ratio_te), 'rb'))
        y_te = pickle.load(open(y_file(DB, fold_te, setting, ratio_te), 'rb'))
    else:
        ifold = (fold_te, fold_val)
        x_val = pickle.load(open(data_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
        y_val = pickle.load(open(y_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
        x_te = pickle.load(open(data_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
        y_te = pickle.load(open(y_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
        x_tr = pickle.load(open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
        y_tr = pickle.load(open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
    
    # conversation en indices 

    intMat = intMat.T
    # get ids of mols and prots for each pairs in validation and test data
    val_label, test_label = y_val, y_te
    val_data, test_data = [], []
    for prot_id, mol_id in x_val:
        val_data.append([dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    val_data = np.stack(val_data, axis=0)
    # val_data is a "nb_samples_in_validation_data * 2" matrix, giving mol_id and prot_id for
    # each pair in the validation data
    for prot_id, mol_id in x_te:
        test_data.append([dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    test_data = np.stack(test_data, axis=0)
    # test_data is a "nb_samples_in_test_data * 2" matrix, giving mol_id and prot_id for
    # each pair in the test data

    # filtre de train 
    # R filtre de W sur intMat

    W = np.zeros(intMat.shape)
    for prot_id, mol_id in x_tr:
        W[dict_mol2ind[mol_id], dict_prot2ind[prot_id]] = 1
    # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
    R = W * intMat

    # if cross validation, find the best parameters on vaidation data
    # else get best parameters in original paper
    if args.cv_val:
        list_param = []
        for r in [50, 100]:
            for x in np.arange(-5, 2):
                for y in np.arange(-5, 3):
                    for z in np.arange(-5, 1):
                        for t in np.arange(-3, 1):
                            list_param.append((r, x, y, z, t))
        list_perf = []
        for par in list_param:
            param = {'c': 5, 'K1': 5, 'K2': 5, 'r': par[0], 'lambda_d': 2**(par[1]),
                     'lambda_t': 2**(par[1]), 'alpha': 2**(par[2]), 'beta': 2**(par[3]),
                     'theta': 2**(par[4]), 'max_iter': 100}
            model = NRLMF(cfix=param['c'], K1=param['K1'], K2=param['K2'],
                          num_factors=param['r'], lambda_d=param['lambda_d'],
                          lambda_t=param['lambda_t'], alpha=param['alpha'],
                          beta=param['beta'], theta=param['theta'],
                          max_iter=param['max_iter'])
            model.pred = np.full(intMat.shape, np.inf)
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            # evaluer la performance en cross validation
            aupr_val, auc_val, _ = model.evaluation(test_data, test_label, intMat, R)
            list_perf.append(aupr_val)

        par = list_param[np.argmax(list_perf)]
        best_param = {'c': 5, 'K1': 5, 'K2': 5, 'r': par[0], 'lambda_d': 2**(par[1]),
                      'lambda_t': 2**(par[1]), 'alpha': 2**(par[2]), 'beta': 2**(par[3]),
                      'theta': 2**(par[4]), 'max_iter': 100}
    else:
        best_param = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125,
                      'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}

    # define model with best parameters, fit and predict
    # regarder a quoi ils correspondent ds le papier
    # de 350 a 380 cross validation pour trouver les paramètres 
    model = NRLMF(cfix=best_param['c'], K1=best_param['K1'], K2=best_param['K2'],
                  num_factors=best_param['r'], lambda_d=best_param['lambda_d'],
                  lambda_t=best_param['lambda_t'], alpha=best_param['alpha'],
                  beta=best_param['beta'], theta=best_param['theta'],
                  max_iter=best_param['max_iter'])

    # taille IntMat
    model.pred = np.full(intMat.shape, np.inf)
    # R = W * intMat
    # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
    # intMat is the binary interaction matrix
    # fit
    model.fix_model(W, intMat, drugMat, targetMat, seed)
    model.predict(test_data, R, intMat)

    # get list of true labels and associated predicted labels for the current test folds
    pred, truth = [], test_label
    for prot_id, mol_id in x_te:
        pred.append(model.pred[dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    pred, truth = np.array(pred), np.array(truth)

    # get performance based on the predictions
    dict_perf = get_clf_perf(pred, truth)
    print(dict_perf)

    # save prediction and performance
    foldname = 'results/pred/' + DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    if not os.path.exists(foldname):
        os.makedirs(foldname)
    pickle.dump((pred, truth, dict_perf), open(foldname + '/NMRLF_' + str(args.cv_val) + '.data',
                                               'wb'))

