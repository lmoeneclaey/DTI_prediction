import argparse
import numpy as np
import os
import pickle

from rdkit import Chem

# errors : "Unable to import ..."
# I think it's due to the fact that we settle "./../DTI_prediction/" as working directory
# but we should find a solution LATER for this pb
# but so far it works, do not care about the error 
from process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from process_dataset.get_molecules_smiles import get_all_DrugBank_smiles
from process_dataset.get_proteins_fastas import get_all_DrugBank_fasta

root = "./../CFTR_PROJECT/"

def process_DB(DB_version, DB_type):
    """ 
    Process the DrugBank database

    The aim is to have the drugs, the proteins and the interactions with \
        these filters:
        - small proteins
        - molecules with know Smiles, loadable with Chem, µM between 100 and 800
        - proteins with all known aa in list, known fasta, and length < 1000

    Use get_all_DrugBank_smiles(), get_all_DrugBank_fasta() and then chek if \
        all the molecules and proteins involved in the interactions respect the\
        filters.

    Writes 11 outputs:
        - [...]_dict_DBid2smiles.data dict keys : DrugBankID values : smile
        - [...]_dict_uniprot2fasta.data dict keys : UniprotID values : fasta
        - [...]_list_interactions.data list [(UniprotID, DrugBankID)]

        - [...]_DBid2smiles.tsv tsv file of the previous data file
        - [...]_uniprot2fasta.tsv tsv file of the previous data file
        - [...]_interactions.tsv tsv file of the previous data file
        
        - [...]_dict_ind2mol.data dict keys : ind values : DrugBankID
        - [...]_dict_mol2ind.data dict keys : DrugBankID values : ind
        - [...]_dict_ind2_prot.data dict keys : ind values : UniprotID
        - [...]_dict_prot2ind.data dict keys : UniprotID values : ind
        - [...]_dict_intMat.npy : matrix of interaction

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" example : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type example : 'S0h'

    Returns
    -------
    None
    """

    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/'

    # create directory
    if not os.path.exists(root + data_dir):
        os.mkdir(root + data_dir)
        print("Directory", DB_type, "for",  DB_version, "created")
    else: 
        print("Directory", DB_type, "for",  DB_version, " already exists")

    dict_id2smile = get_all_DrugBank_smiles(DB_version,
                                            DB_type)
                                            
    dict_uniprot2fasta, list_inter = get_all_DrugBank_fasta(DB_version, 
                                                            DB_type)

    # get the python files of molecule
    dict_id2smile_inter, dict_uniprot2fasta_inter, list_interactions = {}, {}, []
    print('len(list_inter)', len(list_inter))
    for couple in list_inter:
        if couple[1] in list(dict_id2smile.keys()) and \
                couple[0] in list(dict_uniprot2fasta.keys()):
            list_interactions.append(couple)
            dict_id2smile_inter[couple[1]] = dict_id2smile[couple[1]]
            dict_uniprot2fasta_inter[couple[0]] = dict_uniprot2fasta[couple[0]]
    print('nb interactions', len(list_interactions))

    # sorted by DrugBankID
    dict_id2smile_inter_sorted = {}
    for dbid in sorted(dict_id2smile_inter.keys()):
        dict_id2smile_inter_sorted[dbid] = dict_id2smile_inter[dbid]

    # preprocessed directory
    preprocessed_data_dir = root + data_dir + 'preprocessed/'
    os.mkdir(preprocessed_data_dir)

    pickle.dump(dict_id2smile_inter_sorted,
                open(preprocessed_data_dir + DB_type +
                 '_dict_DBid2smiles.data',\
                'wb'))
    pickle.dump(dict_uniprot2fasta_inter,
                open(preprocessed_data_dir + DB_type + 
                '_dict_uniprot2fasta.data',\
                'wb'))
    pickle.dump(list_interactions,
                open(preprocessed_data_dir + DB_type +
                '_list_interactions.data',
                'wb'))

    # tsv files
    dict_ind2prot, dict_prot2ind, dict_ind2mol, dict_mol2ind = {}, {}, {}, {}
    
    f = open(preprocessed_data_dir + DB_type + '_interactions.tsv', 'w')
    for couple in list_interactions:
        f.write(couple[0] + '\t' + couple[1] + '\n')
    f.close()

    f = open(preprocessed_data_dir + DB_type + '_uniprot2fasta.tsv', 'w')
    for ip, prot in enumerate(list(dict_uniprot2fasta_inter.keys())):
        f.write(prot + '\t' + dict_uniprot2fasta_inter[prot] + '\n')
        dict_ind2prot[ip] = prot
        dict_prot2ind[prot] = ip
    f.close()

    f = open(preprocessed_data_dir + DB_type + '_DBid2smiles.tsv', 'w')
    for im, mol in enumerate(list(dict_id2smile_inter_sorted.keys())):
        f.write(mol + '\t' + dict_id2smile_inter_sorted[mol] + '\n')
        dict_ind2mol[im] = mol
        dict_mol2ind[mol] = im
    f.close()

    # Matrix of interactions
    # should not be created 
    intMat = np.zeros((len(list(dict_uniprot2fasta_inter.keys())),
                       len(list(dict_id2smile_inter_sorted.keys()))),
                      dtype=np.int32)
    for couple in list_interactions:
        intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] = 1
    np.save(preprocessed_data_dir + DB_type + '_intMat', intMat)

    # Python files for kernels
    pickle.dump(dict_ind2prot,
                open(preprocessed_data_dir + DB_type +
                '_dict_ind2prot.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_prot2ind,
                open(preprocessed_data_dir + DB_type +
                '_dict_prot2ind.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_ind2mol,
                open(preprocessed_data_dir + DB_type + 
                '_dict_ind2mol.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_mol2ind,
                open(preprocessed_data_dir + DB_type + 
                '_dict_mol2ind.data', 'wb'), 
                protocol=2)

    # save the python object formatted_DB

def get_DB(DB_version, DB_type):
    """ 
    Load the preprocessed DrugBank database
    
    Preprocessed thanks to __main__ in process_DB.py

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
    dict_drug dict keys : DrugBankID values : smile
    dict_protein dict keys : UniprotID values : fasta

    dict_ind2mol dict keys : ind values : DrugBankID
    dict_mol2ind dict keys : DrugBankID values : ind
    dict_ind2_prot dict keys : ind values : UniprotID
    dict_prot2ind dict keys : UniprotID values : ind    
    
    dict_intMat : matrix of interactions
    list_interactions : list of interactions
    """

    # data_dir variables 
    data_dir = 'data/' + DB_version + '/' + DB_type + '/'
    # preprocessed data directory
    preprocessed_data_dir = root + data_dir + 'preprocessed/'

    dict_drug = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_DBid2smiles.data', 'rb'))
    dict_ind2mol = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_ind2mol.data', 'rb'))
    dict_mol2ind = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_mol2ind.data', 'rb'))

    DB_drugs = Drugs(dict_drug = dict_drug,
                     dict_ind2mol = dict_ind2mol,
                     dict_mol2ind = dict_mol2ind)

    dict_protein = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_uniprot2fasta.data', 'rb'))
    dict_ind2prot = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_ind2prot.data', 'rb'))
    dict_prot2ind = pickle.load(open(preprocessed_data_dir + DB_type + 
    '_dict_prot2ind.data', 'rb'))

    DB_proteins = Proteins(dict_protein = dict_protein,
                           dict_ind2prot = dict_ind2prot,
                           dict_prot2ind = dict_prot2ind)

    # intMat = np.load(preprocessed_data_dir + DB_type + '_intMat.npy')
    # ind_inter = np.where(intMat == 1) 

    # Good to create the numpy array at the beginning of the procession of the 
    # DB database
    list_interactions = pickle.load(open(preprocessed_data_dir + DB_type +
                '_list_interactions.data',
                'rb'))
    
    interaction_bool = np.array([1]*len(list_interactions),).reshape(-1, 1)

    # interactions = np.concatenate((np.array(list_interactions), 
    #                               interaction_bool),
    #                               axis = 1)

    DB_couples = Couples(list_couples = list_interactions,
                         interaction_bool = interaction_bool)

    DB = FormattedDB(drugs = DB_drugs,
                     proteins = DB_proteins,
                     couples = DB_couples)

    # return dict_drug,  dict_ind2mol, dict_mol2ind, dict_protein, dict_ind2prot,\
    #      dict_prot2ind, intMat, list_interactions 
    return DB 

if __name__ == "__main__":

    parser = argparse.ArgumentParser("This script processes the DrugBank \
    database in order to have the drugs, the proteins and their \
    interactions with these filters:\n\
    - small proteins\n\
    - molecules with know Smiles, loadable with Chem, µM between 100 and 800\n\
    - proteins with all known aa in list, known fasta, and length < 1000\n")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank version, example: \
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
        # process_DB(args.DB_version,
                #    args.DB_type,
                #    args.process_name)
    
    process_DB(args.DB_version,
               args.DB_type)