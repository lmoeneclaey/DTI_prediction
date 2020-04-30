import pickle
from sklearn.svm import SVC
import numpy as np
import sys
import sklearn

# from process_dataset.process_DB import get_DB

root = './'

#list_m and list_p more than process_DB.get_DB() 
def get_DB(DB='S0'):
    dict_ligand = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_DBid2smiles.data', 'rb'))
    dict_target = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_uniprot2fasta.data', 'rb'))
    intMat = np.load(root + 'data/NNdti_' + DB + '_intMat.npy')
    dict_ind2prot = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_ind2prot.data', 'rb'))
    dict_ind2mol = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_ind2mol.data', 'rb'))
    dict_prot2ind = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_prot2ind.data', 'rb'))
    dict_mol2ind = pickle.load(open(root + 'data/NNdti_' + DB + '_dict_mol2ind.data', 'rb'))
    list_m = [dict_ind2mol[_] for _ in range(len(list(dict_ind2mol.keys())))]
    list_p = [dict_ind2prot[_] for _ in range(len(list(dict_ind2prot.keys())))]

    return list_m, list_p, dict_ligand, dict_target, intMat, \
        dict_ind2prot, dict_ind2mol, dict_prot2ind, dict_mol2ind

# use of split (at the end of the script)
def make_all_K_tr(seed, norm_option, DB='S0', forbidden_list=None):
    list_DB_mol, list_DB_target, dict_DB_mol, dict_DB_target, intMat, \
        dict_ind2prot, dict_ind2mol, dict_prot2ind, dict_mol2ind = get_DB(DB)
    if forbidden_list is not None:
        for couple in forbidden_list:
            if intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] == 0:
                print('attention : forbidden couple ' + str(couple) + ' is already neg')
            else:
                intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] = 0

    if 'unnorm' == norm_option:
        K_mol = pickle.load(open('data/NNdti_' + DB + '_Kmol.data', 'rb'))
        K_prot = pickle.load(open('data/NNdti_' + DB + '_Kprot.data', 'rb'))
    elif 'norm' == norm_option:
        K_mol = pickle.load(open('data/NNdti_' + DB + '_Kmol_norm.data', 'rb'))
        K_prot = pickle.load(open('data/NNdti_' + DB + '_Kprot_norm.data', 'rb'))

    list_couple, y, ind_inter, ind_non_inter = split(intMat, seed, dict_ind2prot, dict_ind2mol)

    nb_couple = len(list_couple)
    print('nb_couple', nb_couple)
    K = np.zeros((nb_couple, nb_couple))
    for i in range(nb_couple):
        ind1_prot = dict_prot2ind[list_couple[i][0]]
        ind1_mol = dict_mol2ind[list_couple[i][1]]
        for j in range(i, nb_couple):
            ind2_prot = dict_prot2ind[list_couple[j][0]]
            ind2_mol = dict_mol2ind[list_couple[j][1]]

            K[i, j] = K_mol[ind1_mol, ind2_mol] * K_prot[ind1_prot, ind2_prot]
            K[j, i] = K[i, j]

    K_tr = K  # Kernels are not centered and normalized !!!!!!!!!
    y_tr = np.array(y)
    list_couples_tr = list_couple

    return K_tr, y_tr, list_couples_tr, ind_inter, ind_non_inter


def test_all_clf(C, DB, norm_option, forbidden_list=None):
    # list_C = [0.01, 0.1, 1., 10., 100., 1000.]
    # list_norm_option = ['norm', 'unnorm']
    seed, n_folds = 91, 10
    np.random.seed(seed)
    # list_seed = np.random.randint(10000, size=n_folds)
    K_all, y_all, list_couples, ind_inter, ind_non_inter = make_all_K_tr(seed, norm_option, DB,
                                                                         forbidden_list)
    # K_all, y_all, list_couples, ind_inter, ind_non_inter = \
    #     np.zeros((1000, 1000)), np.zeros(1000), ['a' for _ in range(1000)], \
    #     np.zeros(500), np.zeros(500)
    # y_all[:500] += 1

    X_ = np.zeros((K_all.shape[0], 1))

    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds,
                                                  shuffle=True, random_state=seed)

    # for iC, C in enumerate(list_C):
    C = C
    print('C', C)
    # for io, norm_option in enumerate(list_norm_option):
    # norm_option = 'norm'
    print('norm_option', norm_option)
    list_aupr, list_auc = [], []

    it = 0
    for tr, te in skf.split(X_, y_all):
        # import pdb; pdb.Pdb().set_trace()
        K_tr, K_te, y_tr, y_te = K_all[tr, :], K_all[:, tr], y_all[tr], y_all[te]
        K_tr, K_te = K_tr[:, tr], K_te[te, :]

        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_tr, y_tr)
        y_pred = clf.decision_function(K_te)
        y_proba = clf.predict_proba(K_te)[:, 1]

        pickle.dump((y_proba, y_te),
                    open('data/clf/' + DB + '_SVM_all_' + norm_option + '_PredValue_' +
                         str(C) + '_' + str(it) + '.data', 'wb'))
        pickle.dump(np.array(list_couples)[te].tolist(),
                    open('data/clf/' + DB + '_SVM_all_' + norm_option + '_PredValue_' +
                         str(C) + '_' + str(it) + '_listcouples.data', 'wb'))

        list_aupr.append(sklearn.metrics.average_precision_score(y_te, y_pred))
        list_auc.append(sklearn.metrics.roc_auc_score(y_te, y_pred))
        # list_acc.append(sklearn.metrics.accuracy_score(y_te, y_pred))
        it += 1
    print('aupr', round(np.mean(list_aupr), 4), round(np.std(list_aupr), 4))
    print('auc', round(np.mean(list_auc), 4), round(np.std(list_auc), 4))
    # print('acc', round(np.mean(list_acc), 4), round(np.std(list_acc), 4))
    print('')
# dict_per_S0_all_clf = {0.01: ,
# 0.1: ,
# 1.: aupr 0.9412 0.0029 auc 0.9227 0.0042,
# 10.: aupr 0.944 0.0027 auc 0.9258 0.0041,
# 100.: aupr 0.9439 0.0027 auc 0.9258 0.0041 ,
# 1000.: aupr 0.9439 0.0027 auc 0.9258 0.0041}
# dict_per_S_all_clf = {0.01: ,
# 0.1: aupr 0.8738 0.0062 auc 0.8662 0.0047,
# 1.: aupr 0.9323 0.003 auc 0.9135 0.0025,
# 10.: aupr 0.938 0.0029 auc 0.9185 0.0027,
# 100.: aupr 0.938 0.0029 auc 0.9185 0.0027,
# 1000.: aupr 0.938 0.0029 auc 0.9185 0.0027 }


def save_all_clf(norm_option='norm', DB='S0', forbidden_list=None):
    C = 10

    print('norm_option', norm_option)
    list_clf, list_couples_of_clf, list_ind_non_inter = [], [], []
    list_seed = [71, 343, 928, 2027, 2]  # , 5309, 554, 55, 1006, 237]
    for seed in list_seed:
        print("seed:", seed)
        K_tr, y_tr, list_couples, ind_inter, ind_non_inter = make_all_K_tr(seed, norm_option, DB,
                                                                           forbidden_list)
        # K_tr, y_tr, list_couples, ind_inter, ind_non_inter = \
        #     np.zeros((1000, 1000)), np.zeros(1000), ['a' for _ in range(1000)], \
        #     np.zeros(500), np.zeros(500)
        y_tr[:500] += 1
        print('training set')
        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_tr, y_tr)
        print('clf done')
        list_clf.append(clf)
        list_couples_of_clf.append(list_couples)
        list_ind_non_inter.append(ind_non_inter)

    for i1 in range(len(list_seed)):
        list_sample1 = [str(list_ind_non_inter[i1][0][_]) + str(list_ind_non_inter[i1][1][_])
                        for _ in range(len(list_ind_non_inter[i1][0]))]
        s = ''
        for i2 in range(len(list_seed)):
            list_sample2 = [str(list_ind_non_inter[i2][0][_]) + str(list_ind_non_inter[i2][1][_])
                            for _ in range(len(list_ind_non_inter[i2][0]))]
            s += str(len(np.intersect1d(list_sample1, list_sample2)))
            s += ', '
        print(s)

    if forbidden_list is not None:
        for inst in forbidden_list:
            DB += '_' + str(inst)
    if norm_option == 'norm':
        clf_filename = 'data/clf/' + DB + '_SVM_all_list_clf_norm.data'
        list_filename = 'data/clf/' + DB + '_SVM_all_list_couples_of_clf_norm.data'
    elif norm_option == 'unnorm':
        clf_filename = 'data/clf/' + DB + '_SVM_all_list_clf.data'
        list_filename = 'data/clf/' + DB + '_SVM_all_list_couples_of_clf.data'
    if norm_option == 'norm':
        clf_filename = 'data/clf/' + DB + '_SVM_all_list_clf_norm.data'
        list_filename = 'data/clf/' + DB + '_SVM_all_list_couples_of_clf_norm.data'
    pickle.dump(list_clf, open(clf_filename, 'wb'))
    pickle.dump(list_couples_of_clf, open(list_filename, 'wb'))

def split(intMat, seed, dict_ind2prot, dict_ind2mol):
    # ind_inter : indices where there is an interaction
    # ind_non_inter : indices where there is not an interaction
    # get list_couples for the seed
    ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
    np.random.seed(seed)
    # choose between all the non-interactions len(list_interactions) couples,
    #  without replacement
    mask = np.random.choice(np.arange(len(ind_non_inter[0])), 
                            len(ind_inter[0]),
                            replace=False)
    # on récupère alors en liste les couples des non interactions
    # must change name of ind_non_inter
    ind_non_inter = (ind_non_inter[0][mask], ind_non_inter[1][mask])
    print("list_on_inter made")
    list_couple, y = [], []
    for i in range(len(ind_inter[0])):
        list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
        y.append(1)
    for i in range(len(ind_non_inter[0])):
        list_couple.append((dict_ind2prot[ind_non_inter[0][i]], dict_ind2mol[ind_non_inter[1][i]]))
        y.append(0)
    print('list couple get')
    return list_couple, y, ind_inter, ind_non_inter


# test_all_clf for S0h, save_all_clf for S0h and forbidden etc, plot_dist_of_pred for S0h,
# make_pred for S0h and forbidden_list etc
if __name__ == '__main__':
    if sys.argv[1] == 'test_all_clf':
        C = float(sys.argv[2])
        DB = sys.argv[3]
        norm_option = sys.argv[4]
        test_all_clf(C, DB, norm_option)
    elif sys.argv[1] == 'save_all_clf':
        norm_option = sys.argv[2]
        DB = sys.argv[3]
        if len(sys.argv) > 4:
            forbidden_list = sys.argv[4].split(',')
            forbidden_list = [(inst[7:], inst[:7]) for inst in forbidden_list]  # DB11920
        else:
            forbidden_list = None
        print('forbidden_list', forbidden_list)
        save_all_clf(norm_option, DB, forbidden_list)

