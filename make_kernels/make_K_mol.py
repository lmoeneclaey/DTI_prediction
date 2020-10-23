import argparse
import math
import numpy as np
import os
import pickle
import sys

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.preprocessing import KernelCenterer

from DTI_prediction.process_dataset.DB_utils import Drugs, FormattedDB
from DTI_prediction.process_dataset.process_DB import get_DB

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

def normalise_kernel(K_temp):
    """ 
    normalise the Kernel matrix

    Parameters
    ----------
    K_temp : numpy array of shape *nb_item*
            Kernel matrix

    Returns
    -------
    K_norm : numpy array of shape *nb_item* normalised Kernel matrix
    """

    nb_item = K_temp.shape[0]
    K_norm = np.zeros((nb_item, nb_item))
    for i in range(nb_item):
        for j in range(i, nb_item):
            K_norm[i, j] = K_temp[i, j] / math.sqrt(K_temp[i, i] * K_temp[j, j])
            K_norm[j, i] = K_norm[i, j]

    return K_norm

# def make_mol_kernel(DB_version, DB_type):
def make_mol_kernel(drugs):
    """ 
    Compute the molecules kernels

    Calculate the ECFP (Morgan fingerprint) for each moleculte and compute the 
    Tanimoto Similarity between all of them.

    Use center_and_normalise_kernel()

    Write 3 files:
        - ..._X_ChemFingerprint.data : numpy array with each line representing \
            the ECFP for each molecule (1024 arguments)
        - ..._K_mol.data : Tanimoto Similarity Kernel
        - ..._K_mol_norm.data : Centered and Normalised Kernel

    Parameters
    ----------
    drugs : Drugs 

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

    dict_drug = drugs.dict_drug
    dict_ind2mol = drugs.dict_ind2mol

    # get the ECFP fingerprints
    nb_mol = drugs.nb
    X_fingerprint = np.zeros((nb_mol, 1024), dtype=np.int32)
    list_fingerprint = []
    # for i in list(dict_ind2mol.keys()):
    for i in range(nb_mol):
        dbid = dict_ind2mol[i]
        m = Chem.MolFromSmiles(dict_drug[dbid])
        list_fingerprint.append(AllChem.GetMorganFingerprint(m, 2))
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 
                                                  2, 
                                                  nBits=1024), 
                                        arr)
        X_fingerprint[i, :] = arr

    # get the Tanimoto Similarity Matrix
    K = np.zeros((len(list_fingerprint), len(list_fingerprint)))
    for i in range(len(list_fingerprint)):
        for j in range(i, len(list_fingerprint)):
            K[i, j] = DataStructs.TanimotoSimilarity(list_fingerprint[i], 
                                                     list_fingerprint[j])
            K[j, i] = K[i, j]

    return X_fingerprint, K

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Calculate the ECFP (Morgan fingerprint) for each moleculte and compute \
    the Tanimoto Similarity between all of them.")

    parser.add_argument("DB_version", type = str, 
                        choices = ["drugbank_v5.1.1", "drugbank_v5.1.5"],
                        help = "the number of the DrugBank version, example: \
                        'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    args = parser.parse_args()

    # data_dir variable 
    data_dir = 'data/' + args.DB_version + '/' + args.DB_type + '/'

    # kernels directory
    kernels_dir = root + data_dir + 'kernels/'
    if not os.path.exists(kernels_dir):
        os.mkdir(kernels_dir)
        print("Kernels directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("Kernels directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    # features directory
    features_dir = root + data_dir + 'features/'
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)
        print("Features directory for", args.DB_type, ",", args.DB_version,
        "created.")
    else:
        print("Features directory for", args.DB_type, ",", args.DB_version,
        "already exists.")

    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    
    X_fingerprint, K = make_mol_kernel(preprocessed_DB.drugs)

    pickle.dump(X_fingerprint,
                open(features_dir + args.DB_type + '_mol_ChemFingerprint.data',
                'wb'))

    # normalized or unnormalized
    for norm_type in ['norm', 'unnorm']:
        if norm_type == 'unnorm':
            kernel_filename = kernels_dir + args.DB_type + '_K_mol.data'
            pickle.dump(K, open(kernel_filename, 'wb'), protocol=2)
        elif norm_type == 'norm':
            K_norm = center_and_normalise_kernel(K)
            kernel_norm_filename = kernels_dir + args.DB_type + '_K_mol_norm.data'
            pickle.dump(K_norm, open(kernel_norm_filename, 'wb'), protocol=2)

    print(K[100, 100], K_norm[100, 100])
    print(K[100, :], K_norm[100, :])
    print(K[500, 500], K_norm[500, 500])
    print(K[500, :], K_norm[500, :])