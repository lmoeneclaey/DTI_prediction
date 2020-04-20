import numpy as np
import pickle
import sys
import math
import os
import argparse

from sklearn.preprocessing import KernelCenterer

from process_DB import get_DB
from make_K_mol import center_and_normalise_kernel

root = './../CFTR_PROJECT/'
# Change to path in my $HOME 
# But problem in compiling LAkernel-0.2 and LAkernel-0.3.2
LAkernel_path = '/cbio/donnees/bplaye/LAkernel-0.2/LAkernel_direct'

def make_temp_Kprot(DB_version, DB_type, process_name, index):
    """ 
    Process the similarity of one particular protein with the others

    Process the similarity (thanks to the L(ocal)A(lignment) Kernel) of \
        the protein (with the key *index* in the dict_ind2prot dictionary, and \
        corresponding to the fasta FASTA_A) with the proteins between *index+1*\
        and *nb_prot* (corresponding to fasta FASTA_B) in the dict_ind2prot \
        dictionary, with the command: \
        'LAkernel_direct FASTA_A FASTA B'

    Then append the output to the file LA_kernel/LA_..._[dbid].txt, where dbid \
        is the value of the key *index* in the dict_ind2prot dictionary. 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'
    index : int
        index of the protein in the dictionaries

    Returns
    -------
    None
    """   
    print("check - make_temp_Kprot")

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'
    
    #create LAkernel directory
    if not os.path.exists(root + data_dir + 'LAkernel/'):
        os.mkdir(root + data_dir + 'LAkernel/')
        print("LAkernel directory for", data_dir, "created")

    # get the DataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]
    dict_ind2prot = preprocessed_DB[3]

    # output_filename
    dbid = dict_ind2prot[index]
    output_filename = root + data_dir + 'LAkernel/LA_' + pattern_name + \
        '_' + dbid + '.txt'


    nb_prot = len(list(dict_target.keys()))
    FASTA1 = dict_target[dbid]
    if not os.path.isfile(output_filename):
        print(index, ":", dbid)
        for j in range(index, nb_prot):
            dbid2 = dict_ind2prot[j]
            FASTA2 = dict_target[dbid2]
            com = LAkernel_path + ' ' + FASTA1 + ' ' + FASTA2 + \
                ' >> ' + output_filename
            cmd = os.popen(com)
            cmd.read()

def make_group_Kprot(DB_version, DB_type, process_name):
    """
    Process the similarity between all the proteins with LAkernel

    Use make_K_mol.center_and_normalise_kernel()

    Write 2 files:
        - ..._Kprot.data : LA Kernel
        - ..._Kprot_norm.data : Centered and Normalised Kernel

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    process_name : str
        string of the process name ex: 'NNdti'

    Returns
    -------
    None
    """

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]
    dict_ind2prot = preprocessed_DB[3]

    nb_prot = len(list(dict_target.keys()))
    X = np.zeros((nb_prot, nb_prot))
    for i in range(nb_prot):


        # output_filename
        dbid = dict_ind2prot[i]
        output_filename = root + data_dir + 'LAkernel/LA_' + pattern_name + \
            '_' + dbid + '.txt'

        j = i
        for line in open(output_filename, 'r'):
            r = float(line.rstrip())
            X[i, j] = r
            X[j, i] = X[i, j]
            j += 1
        if j != nb_prot:
            print(dbid, 'kernel, corresponding to the protein nb ', i, 
            ', is uncompleted')

    # normalized or unnormalized
    for norm_type in ['norm', 'unnorm']:
        if norm_type == 'unnorm':
            kernel_filename = root + data_dir + pattern_name + '_Kprot.data'
            pickle.dump(X, open(kernel_filename, 'wb'), protocol=2)
        elif norm_type == 'norm':
            K_norm = center_and_normalise_kernel(X)
            kernel_filename = root + data_dir + pattern_name + '_Kprot_norm.data'
            pickle.dump(K_norm, open(kernel_filename, 'wb'), protocol=2)

    print(X[100, 100], K_norm[100, 100])
    print(X[100, :], K_norm[100, :])
    print(X[500, 500], K_norm[500, 500])
    print(X[500, :], K_norm[500, :])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Process the similarity between all the proteins with LAkernel")

    # we want to choose between different string for action
    parser.add_argument("action", type = str, choices = ["temp", "group"],
                        help = "if you want to process the similarity to one \
                            (temp) or all of the proteins (group)")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    # need to change for the uniprotID
    parser.add_argument("-i", "--index", type = int,
                        help = "the index of the protein in question \
                        exclusively for make_temp_Kprot()")

    # parser.add_argument("-v", "--verbosity", action = "store_true", 
                        # help = "increase output verbosity")

    args = parser.parse_args()

    if args.action == "temp":
        make_temp_Kprot(args.DB_version, args.DB_type, args.process_name, \
            args.index)

    elif args.action == "group":
        make_group_Kprot(args.DB_version, args.DB_type, args.process_name)