import numpy as np
import pickle
import sys
import math
import argparse

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.preprocessing import KernelCenterer

from process_DB import get_DB

# we have to change it in order to be robust
root = './../CFTR_PROJECT/'
# LIST_AA = ['Q', 'Y', 'R', 'W', 'T', 'F', 'K', 'V', 'S', 'C', 'H', 'L', 'E', \
    # 'P', 'D', 'N', 'I', 'A', 'M', 'G']
# NB_MAX_ATOMS = 105
# MAX_SEQ_LENGTH = 1000
# NB_ATOM_ATTRIBUTES = 32
# NB_BOND_ATTRIBUTES = 8

def center_and_normalise_kernel(K_temp):
    """ 
    Center and normalise the Kernel matrix

    Parameters
    ----------
    K_temp : numpy array of shape *nb_item*
            Kernel matrix

    Returns
    -------
    K_norm : numpy array of shape *nb_item*
            centered and normalised Kernel matrix
    """
    
    K_temp = KernelCenterer().fit_transform(K_temp)
    nb_item = K_temp.shape[0]
    K_norm = np.zeros((nb_item, nb_item))
    for i in range(nb_item):
        for j in range(i, nb_item):
            K_norm[i, j] = K_temp[i, j] / math.sqrt(K_temp[i, i] * K_temp[j, j])
            K_norm[j, i] = K_norm[i, j]

    return K_norm

def make_mol_kernel(DB_version, DB_type, process_name):
    """ 
    Compute the molecules kernels

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
    """

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed

    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_ligand = preprocessed_DB[0]
    dict_ind2mol = preprocessed_DB[4]

    # get the ECFP fingerprints
    nb_mol = len(list(dict_ligand.keys()))
    X_fingerprint = np.zeros((nb_mol, 1024), dtype=np.int32)
    list_fingerprint = []
    for i in list(dict_ind2mol.keys()):
        m = Chem.MolFromSmiles(dict_ligand[dict_ind2mol[i]])
        list_fingerprint.append(AllChem.GetMorganFingerprint(m, 2))
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 
                                                  2, 
                                                  nBits=1024), 
                                        arr)
        X_fingerprint[i, :] = arr
    pickle.dump(X_fingerprint,
                open(root + data_dir + pattern_name + '_X_ChemFingerprint.data',
                'wb'))
    del X_fingerprint

    # get the Tanimoto Similarity Matrix
    X = np.zeros((len(list_fingerprint), len(list_fingerprint)))
    for i in range(len(list_fingerprint)):
        for j in range(i, len(list_fingerprint)):
            X[i, j] = DataStructs.TanimotoSimilarity(list_fingerprint[i], 
                                                     list_fingerprint[j])
            X[j, i] = X[i, j]

    # normalized or unnormalized
    for norm_type in ['norm', 'unnorm']:
        if norm_type == 'unnorm':
            kernel_filename = root + data_dir + pattern_name + '_Kmol.data'
            pickle.dump(X, open(kernel_filename, 'wb'), protocol=2)
        elif norm_type == 'norm':
            K_norm = center_and_normalise_kernel(X)
            kernel_norm_filename = root + data_dir + pattern_name + '_Kmol_norm.data'
            pickle.dump(K_norm, open(kernel_norm_filename, 'wb'), protocol=2)

    print(X[100, 100], K_norm[100, 100])
    print(X[100, :], K_norm[100, :])
    print(X[500, 500], K_norm[500, 500])
    print(X[500, :], K_norm[500, :])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Calculate the ECFP (Morgan fingerprint) for each moleculte and compute \
    the Tanimoto Similarity between all of them.")

    parser.add_argument("DB_version", type = str,
                        help = "the number of the DrugBank version, example: \
                        'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    # parser.add_argument("-v", "--verbosity", action = "store_true", 
                        # help = "increase output verbosity")

    args = parser.parse_args()

    # if args.verbose:
        # print("find something")
    # else:
        # make_mol_kernel(args.DB_version,
                #    args.DB_type,
                #    args.process_name)
    
    make_mol_kernel(args.DB_version, 
                    args.DB_type,
                    args.process_name)