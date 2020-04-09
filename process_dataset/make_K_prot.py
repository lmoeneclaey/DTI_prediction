import numpy as np
import pickle
import sys
import math
import os

from sklearn.preprocessing import KernelCenterer

from process_DB import get_DB
from make_K_mol import center_and_normalise_kernel

root = './'
# LIST_AA = ['Q', 'Y', 'R', 'W', 'T', 'F', 'K', 'V', 'S', 'C', 'H', 'L', 'E', 'P', 'D', 'N',
        #    'I', 'A', 'M', 'G']
# NB_MAX_ATOMS = 105
# MAX_SEQ_LENGTH = 1000
# NB_ATOM_ATTRIBUTES = 32
# NB_BOND_ATTRIBUTES = 8

def make_temp_Kprot(DB_version, DB_type, process_name, index):
    
    '''Process the similarity between one particular protein with the others

    Process the similarity (thanks to the L(ocal)A(lignment) Kernel) between \
        the protein (with the key *index* in the dict_target dictionary, and \
        corresponding to the fasta FASTA_A) with the proteins between *index+1* \
        and *nb_prot* (corresponding to fasta FASTA_B), with the command:
            $HOME/LAkernel-0.2/LAkernel_direct FASTA_A FASTA B

    Then append the output to the file LA_kernel/LA_..._[index].txt 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'

    Returns
    -------
    None

    '''   

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'
    # output_filename
    output_filename = root + data_dir + 'LAkernel/LA_' + pattern_name + \
        '_' + str(index) + '.txt'
    
    # get the DataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]
    dict_ind2prot = preprocessed_DB[3]

    nb_prot = len(list(dict_target.keys()))
    FASTA1 = dict_target[dict_ind2prot[index]]
    if not os.path.isfile(output_filename):
        for j in range(index, nb_prot):
            print(j)
            FASTA2 = dict_target[dict_ind2prot[j]]
            com = '$HOME/LAkernel-0.2/LAkernel_direct ' + FASTA1 + \
                ' ' + FASTA2 + ' >> ' + output_filename
            cmd = os.popen(com)
            cmd.read()


def make_range_temp_Kprot(DB_version, DB_type, process_name, i1, i2):
    '''Process make_temp_Kprot() for a range of proteins

    The proteins got keys between *i1* and *i2* in the dict_target dictionary
    See the description of make_temp_Kprot for more details 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'

    Returns
    -------
    None
    '''

    for index in range(i1, i2):
        make_temp_Kprot(DB_version, DB_type, process_name, index)


# def count_nb_line_in_file(filename):
#     count = None
#     with open(filename, 'r') as f:
#         count = 0
#         for line in f:
#             count += 1
#     return count


def del_temp_Kprot(DB_version, DB_type, process_name):

    '''Process make_temp_Kprot() for a range of proteins

    The proteins got keys between *i1* and *i2* in the dict_target dictionary
    See the description of make_temp_Kprot for more details 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'

    Returns
    -------
    None
    '''

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]

    nb_prot = len(list(dict_target.keys()))

    list_ = []
    for index in range(nb_prot):
        outf = root + data_dir + 'data/LAkernel/LA_' + DB_type + '_' + str(index) + '.txt'
        if os.path.isfile(output_filename):
            count_nb_line = 0
            for line in open(output_filename).xreadlines(  ): count += 1
            # if count_nb_line_in_file(outf) != nb_prot - index:
            if count_nb_line != nb_prot - index:
                list_.append(output_filename)
        else:
            list_.append(output_filename)
    print(list_)

    # for outf in list_:
    #     if os.path.isfile(outf):
    #         os.remove(outf)


def check_temp_Kprot(DB_type):
    '''Check the proteins for which the LAkernel isn't done

    
    '''

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]

    nb_prot = len(list(dict_target.keys()))

    list_ = []
    for index in range(nb_prot):
        outf = root + data_dir + 'LAkernel/LA_' + pattern_name + '_' + str(index) + '.txt'
        if not os.path.isfile(outf):
            list_.append(index)
    print(list_)

    for index in list_:
        make_temp_Kprot(index, DB_type)


def make_group_Kprot(DB_version, DB_type, process_name):

    ''' Compute the proteins kernels

    Calculate the ECFP (Morgan fingerprint) for each moleculte and compute the 
    Tanimoto Similarity between all of them.

    Use center_and_normalise_kernel()

    Write 3 files:
        - ..._X_ChemFingerprint.data : numpy array with each line representing \
            the ECFP for each molecule (1024 arguments)
        - ..._Kmol.data : Tanimoto Similarity Kernel
        - ..._Kmol_norm.data : Centered and Normalised Kernel

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

    Notes
    -----
    Chem.MolFromSmiles(smile) : get the molecule structure from Smiles
    AllChem.GetMorganFingerprint(m, radius) : get the Morgan Fingerprint 
        m : Chem.MolFromSmiles() object
        radius : when 2 roughly equivalent to ECFP4
    '''


    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[1]

    nb_prot = len(list(dict_target.keys()))

    X = np.zeros((nb_prot, nb_prot))
    for i in range(nb_prot):
        j = i
        for line in open(root + data_dir + pattern_name + \
            'LAkernel/LA_' + pattern_name + '_' + str(i) + '.txt', 'r'):
            r = float(line.rstrip())
            X[i, j] = r
            X[j, i] = X[i, j]
            j += 1
        if j != nb_prot:
            print(i, 'not total')

    for norm_type in ['norm', 'unnorm']:
        if norm_type == 'unnorm':
            kernel_filename = root + 'data/NNdti_' + DB_type + '_Kprot.data'
            pickle.dump(X, open(kernel_filename, 'wb'), protocol=2)
        elif norm_type == 'norm':
            K_norm = center_and_normalise_kernel(X)
            kernel_filename = root + 'data/NNdti_' + DB_type + '_Kprot_norm.data'
            pickle.dump(K_norm, open(kernel_filename, 'wb'), protocol=2)

    print(X[100, 100], K_norm[100, 100])
    print(X[100, :], K_norm[100, :])
    print(X[500, 500], K_norm[500, 500])
    print(X[500, :], K_norm[500, :])


if __name__ == "__main__":

    action = sys.argv[1]
    DB_version = sys.argv[2]
    DB_type = sys.argv[3]
    process_name = sys.argv[4]

    if action == "temp":
        index = int(sys.argv[5])
        make_temp_Kprot(DB_version, DB_type, process_name, index)

    elif action == "temp_range":
        i1 = int(sys.argv[5])
        i2 = int(sys.argv[6])
        make_range_temp_Kprot(DB_version, DB_type, process_name, i1, i2)

    elif action == "del":
        del_temp_Kprot(DB_version, DB_type, process_name)

    elif action == "check":
        check_temp_Kprot(DB_version, DB_type, process_name)

    elif action == "group":
        make_group_Kprot(DB_version, DB_type, process_name)