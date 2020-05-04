import argparse
import pickle
import os

from sklearn.svm import SVC

from process_dataset.process_DB import get_DB
from make_K_train import get_list_couples_train, make_K_train
from make_K_inter import get_K_mol_K_prot

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Process the kernel of interactions.")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "where or not to normalize the kernels")

    args = parser.parse_args()

    # pattern_name variable
    pattern_name = args.process_name + '_' + args.DB_type
    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + pattern_name + '/'

    #create directories
    if not os.path.exists(root + 'data/' + args.DB_version + '/' + 'Classifiers'):
        os.mkdir(root + 'data/' + args.DB_version + '/' + 'Classifiers')
        print("Classifiers directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("Classifiers directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    if not os.path.exists(root + 'data/' + args.DB_version + '/' + 'Classifiers/kronSVM'):
        os.mkdir(root + 'data/' + args.DB_version + '/' + 'Classifiers/kronSVM')
        print("kronSVM classifiers directory for ", pattern_name, ", ", args.DB_version,
        "created.")
    else:
        print("kronSVM classifiers directory for ", pattern_name, ", ", args.DB_version,
        "already exists.")

    full_dirname = root + data_dir + 'Classifiers/kronSVM/'

    C = 10.

    preprocessed_DB = get_DB(args.DB_version, args.DB_type, args.process_name)

    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.process_name,
                                 args.norm)

    list_clf, list_couples_of_clf, list_ind_false_inter = [], [], []
    list_seed = [71, 343, 928, 2027, 2]

    for seed in list_seed:
        print("seed:", seed)

        # Create the train dataset
        train_set = get_list_couples_train(seed, preprocessed_DB)
        list_couples = train_set[0]
        y_train = train_set[1]
        ind_false_inter = train_set[3]
        print("Training set prepared.")

        # Compute the kernel of interactions
        # add the list of forbidden
        K_train = make_K_train(seed, preprocessed_DB, kernels) 
        print("Kernel of the interactions computed.")

        # Create the classifier
        clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_train, y_train)
        list_clf.append(clf)
        list_couples_of_clf.append(list_couples)
        list_ind_false_inter.append(ind_false_inter)
        print("Classifiers done.")

        # Classifier name
        couples_filename = full_dirname + pattern_name + \
            '_kronSVM_list_couples_of_clf.data'
        
        if args.norm == True:
            clf_filename = full_dirname + pattern_name + \
            '_kronSVM_list_clf_norm.data'
        else:
            clf_filename = full_dirname + pattern_name + \
            '_kronSVM_list_clf.data'

        pickle.dump(list_couples_of_clf, open(couples_filename, 'wb'))
        pickle.dump(list_clf, open(clf_filename, 'wb'))

        print("Classifiers saved.")