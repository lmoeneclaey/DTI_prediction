import pickle
import numpy as np
import sys
from src.utils.prot_utils import LIST_AA
import collections
import sklearn.model_selection as model_selection
from src.utils.DB_utils import data_file, y_file

def get_DB(DB):

    dict_ligand = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    dict_target = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    intMat = np.load('data/' + DB + '/' + DB + '_intMat.npy')
    dict_ind2prot = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2prot.data', 'rb'))
    dict_ind2mol = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2mol.data', 'rb'))
    dict_prot2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_prot2ind.data', 'rb'))
    dict_mol2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_mol2ind.data', 'rb'))
    return dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind


def DrugBank_CV(DB, list_ratio, setting, cluster_cv, n_folds=5, seed=324):

    # list ratio correspond au ratio nombre d'"interactions inconnues" sur nombre d'
    # "interactions connues".

    # Get the preprocessed DrugBank database
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)

    mratio = max(list_ratio)
    list_ratio = np.sort(list_ratio)[::-1].tolist()
    print('DB got')
    np.random.seed(seed)

    if setting == 1:

        # get the interactions indices
        ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
        n_folds = 5
        np.random.seed(seed)

        # pos folds
        pos_folds_data, pos_folds_y = [], []
        list_couple, y = [], []

        # on peut remplacer list_couple par list_interactions
        # il faudrait ajouter la variable n_interactions
        for i in range(len(ind_inter[0])):
            list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
            y.append(1)
        y, list_couple = np.array(y), np.array(list_couple)

        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92) # pourquoi ne pas mettre seed ici
        skf.get_n_splits(X) # on peut supprimer cette ligne
        ifold = 0

        # pos_folds_data est une liste de 5 (n_folds) listes d'"interactions connues", 
        # chaque liste étant composé de 2931 interactions (2931 * 5 = 14656)
        # pos_folds_y a la meme structure sauf que tous ses éléments sont des 1 
        for train_index, test_index in skf.split(X):
            print(len(train_index), len(test_index))
            pos_folds_data.append(list_couple[test_index].tolist())
            pos_folds_y.append(y[test_index].tolist()) # absurde car ce ne sont que des 1
            ifold += 1
        
        # vérification que les sets de CV sont bien disjoints 
        for n in range(n_folds):
            for n2 in range(n_folds):
                if n2 != n:
                    for c in pos_folds_data[n2]:
                        if c in pos_folds_data[n]:
                            print(c)
                            exit(1)

        # neg folds
        # on choisit parmi toutes les "interactions inconnues" le maximum d'interactions
        # mratio * nb_interactions
        mmask = np.random.choice(np.arange(len(ind_non_inter[0])), len(ind_inter[0]) * mratio,
                                 replace=False)
        ind_non_inter = (ind_non_inter[0][mmask], ind_non_inter[1][mmask])

        # on retrouve les variables list_couple il faudrait changer le terme
        list_couple, y = [], []
        for i in range(len(ind_non_inter[0])):
            list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
                                dict_ind2mol[ind_non_inter[1][i]]))
            y.append(0)
        list_couple, y = np.array(list_couple), np.array(y)

        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)

        # neg_folds_data est un dictionnaire de listes des "interactions inconnues" 
        # pour chaque ratio, les ratio sont alors les keys du dictionnaire
        neg_folds_data, neg_folds_y = {r: [] for r in list_ratio}, {r: [] for r in list_ratio}
        ifold = 0
        for train_index, test_index in skf.split(X):
            neg_folds_data[mratio].append(np.array(list_couple)[test_index].tolist())
            neg_folds_y[mratio].append(np.array(y)[test_index].tolist())
            ifold += 1

        # on remplit les dictionnaires neg_folds_data, neg_folds
        previous_nb_non_inter = len(ind_inter[0]) * mratio
        for ir, ratio in enumerate(list_ratio):
            print(ratio)
            if ratio != mratio:
                nb_non_inter = \
                    round((float(ratio) / (float(list_ratio[ir - 1]))) *
                          previous_nb_non_inter)
                previous_nb_non_inter = nb_non_inter
                nb_non_inter = round(float(nb_non_inter) / float(n_folds))
                print('nb_non_inter', previous_nb_non_inter)
                print(len(neg_folds_data[list_ratio[ir - 1]][0]))
                for ifold in range(n_folds):
                    mask = np.random.choice(
                        np.arange(len(neg_folds_data[list_ratio[ir - 1]][ifold])),
                        nb_non_inter, replace=False)
                    neg_folds_data[ratio].append(
                        np.array(neg_folds_data[list_ratio[ir - 1]][ifold])[mask].tolist())
                    neg_folds_y[ratio].append(
                        np.array(neg_folds_y[list_ratio[ir - 1]][ifold])[mask].tolist())
                    ifold += 1
            print('nb_non_inter', previous_nb_non_inter)

        # save folds
        for ir, ratio in enumerate(list_ratio):
            print("ratio", ratio)
            fo = open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
                      '_folds.txt', 'w')
            folds_data = []
            for ifold in range(n_folds):
                datatemp = pos_folds_data[ifold] + neg_folds_data[ratio][ifold]
                folds_data.append([c[0] + c[1] for c in datatemp])
                ytemp = pos_folds_y[ifold] + neg_folds_y[ratio][ifold]
                # import pdb; pdb.Pdb().set_trace()
                fo.write("ifold " + str(ifold) + '\t' + str(collections.Counter(ytemp)) + '\n')
                pickle.dump(datatemp, open(data_file(DB, ifold, setting, ratio), 'wb'))
                pickle.dump(ytemp, open(y_file(DB, ifold, setting, ratio), 'wb'))
                print("ifold " + str(ifold), str(collections.Counter(ytemp)))
            fo.close()
            for n in range(n_folds):
                for n2 in range(n_folds):
                    if n2 != n:
                        for c in folds_data[n2]:
                            if c in folds_data[n]:
                                print('alerte', c)
                                exit(1)


if __name__ == "__main__":
    DB = sys.argv[1]

    if sys.argv[2] == 'make_CV':
        list_ratio = [10, 5, 2, 1]
        # for setting in [1, 2, 3, 4]:
        for setting in [2]:
            print('setting', setting)
            # if not ('DrugBankH' in sys.argv[1] and ratio == 5 and setting == 1):
            DrugBank_CV(DB, list_ratio, setting, cluster_cv=False)

    if sys.argv[2] == 'check_CV':
        for setting in [1]:
            print("setting", setting)
            for ratio_te in [5, 2, 1]:
                print("ratio_te", ratio_te)
                nfolds = 25 if setting == 4 else 5
                for ite in range(nfolds):
                    print("ite", ite)
                    list_couple_te = pickle.load(open(data_file(DB, ite, setting, ratio_te), 'rb'))
                    list_prot_te = [c[0] for c in list_couple_te]
                    list_mol_te = [c[1] for c in list_couple_te]
                    list_couple_te = [c[0] + c[1] for c in list_couple_te]
                    for ratio_tr in [5, 2, 1]:
                        print("ratio_tr", ratio_tr)
                        for itr in range(nfolds):
                            if itr != ite:
                                print('itr', itr)
                                list_couple_tr = \
                                    pickle.load(open(data_file(DB, itr, setting, ratio_tr), 'rb'))
                                # list_couple_tr = [c[0] + c[1] for c in list_couple_tr]
                                list_y = pickle.load(open(y_file(DB, itr, setting, ratio_tr),
                                                          'rb'))
                                print(len(list_couple_tr))
                                for ic, couple in enumerate(list_couple_tr):
                                    # import pdb; pdb.Pdb().set_trace()
                                    if couple[0] + couple[1] in list_couple_te:
                                        print('ALERTE ', couple, list_y[ic])
                                        exit(1)
                                    if setting in [2, 4] and couple[0] in list_prot_te:
                                        print('alerte mol', couple[1])
                                        exit(1)
                                    elif setting in [3, 4] and couple[1] in list_mol_te:
                                        print('alerte mol', couple[0])
                                        exit(1)

        for setting in [4]:
            print("setting", setting)
            for ratio_te in [5, 2, 1]:
                print("ratio_te", ratio_te)
                nfolds = 25
                for ite in range(nfolds):
                    print("ite", ite)
                    for ival in range(16):
                        print("ival", ival)
                        ifold = (ite, ival)

                        list_couple_te = pickle.load(
                            open(data_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
                        list_prot_te = [c[0] for c in list_couple_te]
                        list_mol_te = [c[1] for c in list_couple_te]
                        list_couple_te = [c[0] + c[1] for c in list_couple_te]

                        list_couple_val = pickle.load(
                            open(data_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
                        list_prot_val = [c[0] for c in list_couple_val]
                        list_mol_val = [c[1] for c in list_couple_val]
                        list_couple_val = [c[0] + c[1] for c in list_couple_val]
                        for ratio_tr in [5, 2, 1]:
                            print("ratio_tr", ratio_tr)
                            list_couple_tr = pickle.load(
                                open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
                            # list_couple_tr = [c[0] + c[1] for c in list_couple_tr]
                            list_y = pickle.load(
                                open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
                            print(len(list_couple_tr), list_couple_tr)
                            for ic, couple in enumerate(list_couple_tr):
                                import pdb; pdb.Pdb().set_trace()
                                if couple[0] + couple[1] in list_couple_te or \
                                        couple[0] + couple[1] in list_couple_val:
                                    print('ALERTE ', couple, list_y[ic])
                                    exit(1)
                                if couple[0] in list_prot_te or couple[0] in list_prot_val:
                                    print('alerte mol', couple[0])
                                    exit(1)
                                elif couple[1] in list_mol_te or couple[1] in list_mol_val:
                                    print('alerte mol', couple[1])
                                    exit(1)