import numpy as np
import pickle
import sys
import math
import os
import argparse

from sklearn.preprocessing import KernelCenterer

from DTI_prediction.process_dataset.DB_utils import Proteins, FormattedDB
from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.make_K_mol import center_and_normalise_kernel

root = './../CFTR_PROJECT/'
# Change to path in my $HOME 
# But problem in compiling LAkernel-0.2 and LAkernel-0.3.2
LAkernel_path = '/cbio/donnees/bplaye/LAkernel-0.2/LAkernel_direct'

def make_temp_K_prot(DB_version, DB_type, index):
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
    index : int
        index of the protein in the dictionaries

    Returns
    -------
    None
    """   

    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/'

    # kernels directory
    kernels_dir = root + data_dir + 'kernels/'

    #create LAkernel directory
    if not os.path.exists(kernels_dir + 'LAkernel/'):
        os.mkdir(kernels_dir + 'LAkernel/')
        print("LAkernel directory for", data_dir, "created")

    # get the DataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type)
    dict_protein = preprocessed_DB.proteins.dict_protein
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot

    # output_filename
    dbid = dict_ind2prot[index]
    output_filename = kernels_dir + 'LAkernel/LA_' + DB_type + \
        '_' + dbid + '.txt'

    nb_prot = preprocessed_DB.proteins.nb
    FASTA1 = dict_protein[dbid]
    if not os.path.isfile(output_filename):
        print(index, ":", dbid)
        for j in range(index, nb_prot):
            dbid2 = dict_ind2prot[j]
            FASTA2 = dict_protein[dbid2]
            com = LAkernel_path + ' ' + FASTA1 + ' ' + FASTA2 + \
                ' >> ' + output_filename
            cmd = os.popen(com)
            cmd.read()
        print("completed")

def make_group_K_prot(DB_version, DB_type):
    """
    Process the similarity between all the proteins with LAkernel

    Use make_K_mol.center_and_normalise_kernel()

    Write 2 files:
        - ..._K_prot.data : LA Kernel
        - ..._K_prot_norm.data : Centered and Normalised Kernel

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type

    Returns
    -------
    None
    """

    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/'

    # kernels directory
    kernels_dir = root + data_dir + 'kernels/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type)
    dict_ind2prot = preprocessed_DB.proteins.dict_ind2prot

    nb_prot = preprocessed_DB.proteins.nb
    X = np.zeros((nb_prot, nb_prot))
    for i in range(nb_prot):


        # output_filename
        dbid = dict_ind2prot[i]
        output_filename = kernels_dir + 'LAkernel/LA_' + DB_type + \
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
            kernel_filename = kernels_dir + DB_type + '_K_prot.data'
            pickle.dump(X, open(kernel_filename, 'wb'), protocol=2)
        elif norm_type == 'norm':
            K_norm = center_and_normalise_kernel(X)
            kernel_filename = kernels_dir + DB_type + '_K_prot_norm.data'
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

    # need to change for the uniprotID
    parser.add_argument("-i", "--index", type = int,
                        help = "the index of the protein in question \
                        exclusively for make_temp_K_prot()")

    # parser.add_argument("-v", "--verbosity", action = "store_true", 
                        # help = "increase output verbosity")

    args = parser.parse_args()

    if args.action == "temp":
        make_temp_K_prot(args.DB_version, 
                         args.DB_type, 
                         args.index)

    elif args.action == "group":
        make_group_K_prot(args.DB_version, 
                          args.DB_type)