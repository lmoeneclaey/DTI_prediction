import pickle
from sklearn.externals import joblib
from sklearn.svm import SVC
from src.make_folds_DB import get_DB
from src.baselines.SVM_on_handcrafted_features import make_Kcouple


if __name__ == "__main__":
    DB = "DrugBankH"
    nb_sampling = 10
    C = 10.

    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)

    Kmol = pickle.load(open('mol_kernels/' + DB + '_Kmol.data', 'rb'))
    Kprot = pickle.load(open('prot_kernels/' + DB + '_Kprot.data', 'rb'))

    for i_sampling in range(nb_sampling):
        print(i_sampling)
        x = pickle.load(open('CLASSIFIERS/data/X_all_' + str(i_sampling) + '.data', 'rb'))
        y = pickle.load(open('CLASSIFIERS/data/y_all_' + str(i_sampling) + '.data', 'rb'))

        print("make Kcouple")
        K = make_Kcouple(x, None, Kmol, Kprot, dict_prot2ind, dict_mol2ind)

        print("fit")
        ml = SVC(kernel='precomputed', probability=True, class_weight='balanced', C=C)
        ml.fit(K, y)

        print("save clf")
        joblib.dump(ml, 'CLASSIFIERS/kronSVM_clf/kronSVM_' + str(i_sampling) + '.sav')