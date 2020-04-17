import numpy as np
from rdkit import Chem
import pandas as pd
import pickle
import sys
import os
import re
import argparse

# maybe to remove
import csv

# TO D0 : split the file between molecules and proteins
# TO DO : order the different dictionaries

# we have to change it in order to be robust
root = "./../CFTR_PROJECT/"
LIST_AA = ['Q', 'Y', 'R', 'W', 'T', 'F', 'K', 'V', 'S', 'C', 'H', 'L', 'E', \
    'P', 'D', 'N', 'I', 'A', 'M', 'G']
# NB_MAX_ATOMS = 105
# MAX_SEQ_LENGTH = 1000
# NB_ATOM_ATTRIBUTES = 32
# NB_BOND_ATTRIBUTES = 8


def check_mol_weight(DB_type, m, dict_id2smile, weight, smile, dbid):
    """ 
    Put a threshold on the molecules' weight
    
    This function is a complete dependency of get_all_DrugBanksmiles() and
    cannot be understood without.
    If it tackles an known Smile with a molecule loadable with Chem AND 
    if DB_type is 'S', it will keep if it is between 100 µM and 800 µM
    if DB_type is 'S0' or 'S0h' do nothing

    Parameters
    ----------
    DB_type : str
              string of the DrugBank type
    m : rdkit.Chem.rdchem.Mol
    dict_id2smile : dictionary
    weight : int
    smile : str
            string of the mol smile

    Returns
    -------
    dict_id2smile : dictionary
        keys : DrugBankId
        values : Smiles
    """ 

    # TO DO : put 'if loop' of 'm is not None and smile != '' before to better 
    # clarity

    if DB_type == 'S':
        if weight > 100 and weight < 800 and m is not None and smile != '':
            dict_id2smile[dbid] = smile
    elif DB_type == 'S0' or DB_type == 'S0h':
        if m is not None and smile != '':
            dict_id2smile[dbid] = smile
    return dict_id2smile

def get_all_DrugBank_smiles(DB_version, DB_type):
    """ 
    Get the smiles of the DrugBank molecules

    Open the file 'structures.sdf' in the data folder. (See description in data.pdf)
    Look at each line and see if it concerns a DrugBank ID, a Smile, 
    or a Molecular Weight (needed in the threshold function check_mol_weight())
    If it does, then it will get the corresponding data from the next line. 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "vX.X.X" exemple : "v5.1.1"
    DB_type : str
              string of the DrugBank type

    Returns
    -------
    dict_id2smile_sorted : dictionary
        keys : DrugBankID
        values : Smiles

    Notes
    -----
    Chel.MolFromSmiles() : get the molecule structure from Smiles
    """ 

    dict_id2smile = {}
    dbid, smile = None, None
    found_id, found_smile, found_weight = False, False, False

    raw_data_dir = 'data/' + DB_version + '/raw/'

    f = open(root + raw_data_dir + 'structures.sdf', 'r')
    for line in f:
        line = line.rstrip() 
        if found_id:
            if smile is not None:
                m = Chem.MolFromSmiles(smile)
                dict_id2smile = check_mol_weight(DB_type, 
                                                m, 
                                                dict_id2smile, 
                                                weight, 
                                                smile, 
                                                dbid)
            dbid = line
            found_id = False
        if found_smile:
            smile = line
            found_smile = False
        if found_weight:
            weight = float(line)
            found_weight = False

        if line == "> <DATABASE_ID>":
            found_id = True
        elif line == "> <SMILES>":
            found_smile = True
        elif line == '> <MOLECULAR_WEIGHT>':
            found_weight = True
    f.close()

    # for the last molecule 
    m = Chem.MolFromSmiles(smile)
    dict_id2smile = check_mol_weight(DB_type, 
                                    m,
                                    dict_id2smile,
                                    weight, 
                                    smile,
                                    dbid)

    # sorted by DrugBankID
    dict_id2smile_sorted = {}
    for dbid in sorted(dict_id2smile.keys()):
        dict_id2smile_sorted[dbid] = dict_id2smile[dbid]

    return dict_id2smile_sorted

def get_specie_per_uniprot(DB_version):
    """ 
    Get the proteins species 

    Open the file 'drugbank_small_molecule_target_polypeptide_ids.csv\
        /all.csv' in the raw data folder. (See descritpion in data.pdf)
    Look at each line (so protein), and get its species.
    The ouptut is necessary for the function check_prot_length().

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "vX.X.X" exemple : "v5.1.1"

    Returns
    -------
    dict_specie_per_prot : dictionary
        keys : DrugBankID
        values : Species

    Notes
    -----
    row[5] = Uniprot ID
    row[11] = Species
    """ 

    raw_data_dir = 'data/' + DB_version + '/raw/'

    dict_specie_per_prot = {}

    # # 1 - using csv_reader

    # reader = \
    #     csv.reader(open(root + raw_data_dir + \
    #         'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv', 'r'),
    #                delimiter=',')
    # i = 0
    # # changer 
    # for row in reader:
    #     # if i > 0:
    #     dict_specie_per_prot[row[5]] = row[11]
    #     # i += 1

    # 2 - using pandas

    df = pd.read_csv(root + raw_data_dir + \
        'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv', sep=',')
    df = df.fillna('')

    df_uniprot_id = df['UniProt ID']
    df_species = df['Species']

    for line in range(df.shape[0]):
        dict_specie_per_prot[df_uniprot_id[line]] = df_species[line]

    # 3 - quicker method

    # df_tronc = df[['UniProt ID', 'Species']]
    # trans_df_tronc = df_tronc.set_index("UniProt ID").T
    # dict_specie_per_prot = trans_df_tronc.to_dict("list")

    return dict_specie_per_prot

def check_prot_length(DB_version, DB_type, fasta, dict_uniprot2seq, dbid, \
    list_ligand, list_inter):
    """ 
    Put a threshold on the molecules' weight
    
    This function is a complete dependency of get_all_DrugBank_fasta() and
    cannot be understood without.
    If DB_type is 'S', it checks if all the aas exist, and the length is less \
        than 1000
    If DB_type is 'S0', do nothing
    If DB_type is 'S0h', it checks if the protein's species is human (thanks \
        to get_specie_per_uniprot())

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "vX.X.X" exemple : "v5.1.1"
    DB_type : str
              string of the DrugBank type
    fasta : str
            sequence of aa
    dict_uniprot2seq : dictionary
    dbid : str
    list_ligand : list
    list_inter : list

    Returns
    -------
    dict_uniprot2seq : dictionary
        keys : UniprotID
        values : Fasta
    list_inter : list 
        [(UniprotID, DrugBankID)]
    """ 
    
    if DB_type == 'S':
        aa_bool = True
        for aa in fasta:
            if aa not in LIST_AA:
                aa_bool = False
                break
        if len(fasta) < 1000 and aa_bool is True:
            dict_uniprot2seq[dbid] = fasta
            for ligand in list_ligand:
                list_inter.append((dbid, ligand))
    elif DB_type == 'S0':
        dict_uniprot2seq[dbid] = fasta
        for ligand in list_ligand:
            list_inter.append((dbid, ligand))
    elif DB_type == 'S0h':
        dict_specie_per_prot = get_specie_per_uniprot(DB_version)
        if dict_specie_per_prot[dbid] == 'Humans':
            dict_uniprot2seq[dbid] = fasta
            for ligand in list_ligand:
                list_inter.append((dbid, ligand))
    return list_inter, dict_uniprot2seq


def get_all_DrugBank_fasta(DB_version, DB_type):
    """ 
    Get the fasta of the Drug Bank proteins

    Open the file 'drugbank_small_molecule_target_polypeptide_sequences.fasta\
        /protein.fasta' in the data folder. (See description in data.pdf)
    Look at each line and see wether it is the description, then it gets the \
        DrangBankID(UniprotID)and the list of ligands DrankBankIDs, wether it \
        is the sequence, then it checks the length (with check_prot_length()). 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "vX.X.X" exemple : "v5.1.1"
    DB_type : str
        string of the DrugBank type

    Returns
    -------
    dict_uniprot2seq_sorted : dictionary
        keys : UniprotID
        values : Fasta
    list_inter_sorted : list 
        [(UniprotID, DrugBankID)]
    """ 

    raw_data_dir = 'data/' + DB_version + '/raw/'

    # put here the data folder argument
    f = open(root + raw_data_dir + \
        'drugbank_small_molecule_target_polypeptide_sequences.fasta/protein.fasta','r')
    dict_uniprot2seq, list_inter = {}, []
    dbid, fasta = None, None

    for line in f:
        
        line = line.rstrip()

        if len(line)>0:
            if line[0] != '>':
                fasta += line
            else:
                if fasta is not None:
                    list_inter, dict_uniprot2seq = check_prot_length(DB_version,
                                                                    DB_type, 
                                                                    fasta, 
                                                                    dict_uniprot2seq,
                                                                    dbid,
                                                                    list_ligand,
                                                                    list_inter)
                dbid = line.split('|')[1].split(' ')[0]
                # list_ligand = line.split('(')[1].split(')')[0].split('; ')
                ligands = re.search("\(([DB0-9; ]*)\)\Z", line)
                ligands_str = ligands.group(1)
                list_ligand = ligands_str.split("; ")
                if type(list_ligand) is not list:
                    list_ligand = [list_ligand]
                fasta = ''
    
    f.close()

    # for the last protein
    list_inter, dict_uniprot2seq = check_prot_length(DB_version,
                                                    DB_type, 
                                                    fasta,
                                                    dict_uniprot2seq, 
                                                    dbid, 
                                                    list_ligand, 
                                                    list_inter)

    # sorted by UniprotID
    dict_uniprot2seq_sorted = {}
    for dbid in sorted(dict_uniprot2seq.keys()):
        dict_uniprot2seq_sorted[dbid] = dict_uniprot2seq[dbid]

    # sorted by UniProtID then DrugBankID
    list_inter_sorted = \
        sorted(list_inter, key=lambda dbid: (dbid[0], dbid[1]))

    return dict_uniprot2seq_sorted, list_inter_sorted


def process_DB(DB_version, DB_type, process_name):
    """ 
    Process the DrugBank database

    The aim is to have the molecules, the proteins and the interactions with \
        these filters:
        - small molecules targets
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

    # create directory
    if not os.path.exists(root + 'data/' + DB_version + '/' + pattern_name):
        os.mkdir(root + 'data/' + DB_version + '/' + pattern_name)
        print("Directory", pattern_name, "for",  DB_version, "created")
    else: 
        print("Directory", pattern_name, "for",  DB_version, " already exists")

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

    pickle.dump(dict_id2smile_inter_sorted,
                open(root + data_dir + pattern_name +
                 '_dict_DBid2smiles.data',\
                'wb'))
    pickle.dump(dict_uniprot2fasta_inter,
                open(root + data_dir + pattern_name + 
                '_dict_uniprot2fasta.data',\
                'wb'))
    pickle.dump(list_interactions,
                open(root + data_dir + pattern_name +
                '_list_interactions.data',
                'wb'))

    # tsv files
    dict_ind2prot, dict_prot2ind, dict_ind2mol, dict_mol2ind = {}, {}, {}, {}
    
    f = open(root + data_dir + pattern_name + '_interactions.tsv', 'w')
    for couple in list_interactions:
        f.write(couple[0] + '\t' + couple[1] + '\n')
    f.close()

    f = open(root + data_dir + pattern_name + '_uniprot2fasta.tsv', 'w')
    for ip, prot in enumerate(list(dict_uniprot2fasta_inter.keys())):
        f.write(prot + '\t' + dict_uniprot2fasta_inter[prot] + '\n')
        dict_ind2prot[ip] = prot
        dict_prot2ind[prot] = ip
    f.close()

    f = open(root + data_dir + pattern_name + '_DBid2smiles.tsv', 'w')
    for im, mol in enumerate(list(dict_id2smile_inter_sorted.keys())):
        f.write(mol + '\t' + dict_id2smile_inter_sorted[mol] + '\n')
        dict_ind2mol[im] = mol
        dict_mol2ind[mol] = im
    f.close()

    # Matrix of interactions
    intMat = np.zeros((len(list(dict_uniprot2fasta_inter.keys())),
                       len(list(dict_id2smile_inter_sorted.keys()))),
                      dtype=np.int32)
    for couple in list_interactions:
        intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] = 1
    np.save(root + data_dir + pattern_name + '_intMat', intMat)

    # Python files for kernels
    pickle.dump(dict_ind2prot,
                open(root + data_dir + pattern_name +
                '_dict_ind2prot.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_prot2ind,
                open(root + data_dir + pattern_name +
                '_dict_prot2ind.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_ind2mol,
                open(root + data_dir + pattern_name + 
                '_dict_ind2mol.data', 'wb'), 
                protocol=2)
    pickle.dump(dict_mol2ind,
                open(root + data_dir + pattern_name + 
                '_dict_mol2ind.data', 'wb'), 
                protocol=2)


def get_DB(DB_version, DB_type, process_name):
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
    dict_ligand dict keys : DrugBankID values : smile
    dict_target dict keys : UniprotID values : fasta

    dict_intMat : matrix of interaction

    dict_ind2mol dict keys : ind values : DrugBankID
    dict_mol2ind dict keys : DrugBankID values : ind
    dict_ind2_prot dict keys : ind values : UniprotID
    dict_prot2ind dict keys : UniprotID values : ind    
    """

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variables 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    dict_ligand = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_DBid2smiles.data', 'rb'))
    dict_target = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_uniprot2fasta.data', 'rb'))

    intMat = np.load(root + data_dir + pattern_name + '_intMat.npy')
    
    dict_ind2mol = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_ind2mol.data', 'rb'))
    dict_mol2ind = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_mol2ind.data', 'rb'))
    dict_prot2ind = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_prot2ind.data', 'rb'))
    dict_ind2prot = pickle.load(open(root + data_dir + pattern_name + 
    '_dict_ind2prot.data', 'rb'))

    return dict_ligand, dict_target, intMat, dict_ind2prot, \
         dict_ind2mol, dict_prot2ind, dict_mol2ind

if __name__ == "__main__":

    parser = argparse.ArgumentParser("This script processes the DrugBank \
database in order to have the molecules, the proteins and their \
interactions with these filters:\n\
    - small molecules targets\n\
    - molecules with know Smiles, loadable with Chem, µM between 100 and 800\n\
    - proteins with all known aa in list, known fasta, and length < 1000\n")

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
        # process_DB(args.DB_version,
                #    args.DB_type,
                #    args.process_name)
    
    process_DB(args.DB_version,
               args.DB_type,
               args.process_name)