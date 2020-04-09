

def get_DB_withK(DB='S0', norm_option='norm'):
    list_DB_mol, list_DB_target, dict_DB_mol, dict_DB_target, intMat, \
        dict_ind2prot, dict_ind2mol, dict_prot2ind, dict_mol2ind = get_DB(DB)
    if 'unnorm' == norm_option:
        K_mol = pickle.load(open('data/NNdti_' + DB + '_Kmol.data', 'rb'))
        K_prot = pickle.load(open('data/NNdti_' + DB + '_Kprot.data', 'rb'))
    elif 'norm' == norm_option:
        K_mol = pickle.load(open('data/NNdti_' + DB + '_Kmol_norm.data', 'rb'))
        K_prot = pickle.load(open('data/NNdti_' + DB + '_Kprot_norm.data', 'rb'))
    return list_DB_mol, list_DB_target, dict_DB_mol, dict_DB_target, intMat, dict_ind2prot, \
        dict_ind2mol, dict_prot2ind, dict_mol2ind, K_mol, K_prot