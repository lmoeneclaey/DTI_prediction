import pickle
from sklearn.svm import SVC
import numpy as np
import sys, os
from src.process_dataset.make_S0_SVM_all_clf import get_DB_withK, get_DB, load_nei_dict, make_nei_train


root = './'


def make_all_K_te(instance, list_couples, norm_option, DB):
    list_DB_mol, list_DB_target, dict_DB_mol, dict_DB_target, intMat, \
        dict_ind2prot, dict_ind2mol, dict_prot2ind, dict_mol2ind = get_DB(DB)
    if 'unnorm' == norm_option:
        K_mol = pickle.load(open(root + 'data/NNdti_' + DB + '_Kmol.data', 'rb'))
        K_prot = pickle.load(open(root + 'data/NNdti_' + DB + '_Kprot.data', 'rb'))
    elif 'norm' == norm_option:
        K_mol = pickle.load(open(root + 'data/NNdti_' + DB + '_Kmol_norm.data', 'rb'))
        K_prot = pickle.load(open(root + 'data/NNdti_' + DB + '_Kprot_norm.data', 'rb'))

    print(instance)
    if instance[:2] == 'DB':
        K_te = np.zeros((len(list_DB_target), len(list_couples)))
        for i_prot, prot in enumerate(list_DB_target):
            for j_couple, couple in enumerate(list_couples):
                K_te[i_prot, j_couple] = \
                    K_mol[dict_mol2ind[instance], dict_mol2ind[couple[1]]] * \
                    K_prot[dict_prot2ind[prot], dict_prot2ind[couple[0]]]
    else:
        K_te = np.zeros((len(list_DB_mol), len(list_couples)))
        for i_mol, mol in enumerate(list_DB_mol):
            for j_couple, couple in enumerate(list_couples):
                K_te[i_mol, j_couple] = \
                    K_mol[dict_mol2ind[mol], dict_mol2ind[couple[1]]] * \
                    K_prot[dict_prot2ind[instance], dict_prot2ind[couple[0]]]
    return K_te


def make_prediction_all(instance, norm_option, DB, forbidden_list):
    list_DB_mol, list_DB_target, dict_DB_mol, dict_DB_target, intMat, \
        dict_ind2prot, dict_ind2mol, dict_prot2ind, dict_mol2ind = get_DB(DB)

    if forbidden_list is not None:
        DB_str = DB
        for inst in forbidden_list:
            DB_str += '_' + str(inst)
    else:
        DB_str = DB
    print(DB_str)
    if 'unnorm' == norm_option:
        list_clf = pickle.load(open(root + 'data/clf/' + DB_str + '_SVM_all_list_clf.data', 'rb'))
        filename = root + 'data/clf/pred_SVM_all/' + DB_str + '_'
        list_couples_of_clf = \
            pickle.load(open(root + 'data/clf/' + DB_str + '_SVM_all_list_couples_of_clf.data',
                             'rb'))
    elif 'norm' == norm_option:
        list_clf = pickle.load(open(root + 'data/clf/' + DB_str + '_SVM_all_list_clf_norm.data',
                                    'rb'))
        filename = root + 'data/clf/pred_SVM_all_norm/' + DB_str + '_'
        list_couples_of_clf = \
            pickle.load(open(root + 'data/clf/' + DB_str +
                             '_SVM_all_list_couples_of_clf_norm.data', 'rb'))

    if not os.path.isfile(filename + instance + '.data'):
        print('instance', instance)
        if instance[:2] == 'DB':
            pred = np.zeros((len(list_DB_target), len(list_clf)))
        else:
            pred = np.zeros((len(list_DB_mol), len(list_clf)))
        for ind in range(len(list_clf)):
            K_te = make_all_K_te(instance, list_couples_of_clf[ind], norm_option, DB)
            print('K_te done')
            pred[:, ind] = list_clf[ind].predict_proba(K_te)[:, 1]

        # pred_avg, pred_std = np.round(np.mean(pred, axis=1), 4), np.round(np.std(pred, axis=1), 4)
        # pred_ind = np.argsort(pred_avg)[::-1]

        pickle.dump(pred, open(filename + instance + '.data', 'wb'))

# DB08820, DB11712, DB09280
if __name__ == '__main__':

    instance = sys.argv[2]
    if len(sys.argv) > 3:
        forbidden_list = sys.argv[3].split(',')
        forbidden_list = [(inst[7:], inst[:7]) for inst in forbidden_list]  # DB11920
    else:
        forbidden_list = None
    if 'unnorm' in sys.argv[1]:
        norm_option = 'unnorm'
    elif 'norm' in sys.argv[1]:
        norm_option = 'norm'
    if 'S0h' in sys.argv[1]:
        DB = 'S0h'
    elif 'S0' in sys.argv[1]:
        DB = 'S0'
    elif 'S' in sys.argv[1]:
        DB = 'S'

    if 'all' in sys.argv[1]:
        print('all', DB, norm_option)
        make_prediction_all(instance, norm_option, DB, forbidden_list)