import numpy as np
import pickle

root = "./../CFTR_PROJECT/"

def get_K_mol_K_prot(DB_version, DB_type, norm):
    """ 
    Load the molecules and the proteins kernels

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type
    norm : boolean
        normalized or unnormalized
        "normalized" by default

    Returns
    -------
    K_mol
    K_prot
    """   

    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/'
    # kernels directory
    kernels_dir = root + data_dir + 'kernels/'

    if norm == False:
        K_mol = pickle.load(open(kernels_dir + DB_type + '_K_mol.data', 'rb'))
        K_prot = pickle.load(open(kernels_dir + DB_type + '_K_prot.data', 'rb'))
    elif norm == True:
        K_mol = pickle.load(open(kernels_dir + DB_type + '_K_mol_norm.data', 'rb'))
        K_prot = pickle.load(open(kernels_dir + DB_type + '_K_prot_norm.data', 'rb'))

    return K_mol, K_prot