import pandas as pd
import re

# maybe to remove
import csv

# we have to change it in order to be robust
root = "./../CFTR_PROJECT/"
LIST_AA = ['Q', 'Y', 'R', 'W', 'T', 'F', 'K', 'V', 'S', 'C', 'H', 'L', 'E', \
    'P', 'D', 'N', 'I', 'A', 'M', 'G']
# NB_MAX_ATOMS = 105
# MAX_SEQ_LENGTH = 1000
# NB_ATOM_ATTRIBUTES = 32
# NB_BOND_ATTRIBUTES = 8

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

        aa_bool = True
        for aa in fasta:
            if aa not in LIST_AA:
                aa_bool = False
                break
        if (aa_bool ==True and dict_specie_per_prot[dbid] == 'Humans'):
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

    # list_inter got some duplicates due to the fasta file
    list_inter_unique = set(list_inter)
    # list_inter sorted by UniProtID then DrugBankID
    list_inter_sorted = \
        sorted(list_inter_unique , key=lambda dbid: (dbid[0], dbid[1]))

    return dict_uniprot2seq_sorted, list_inter_sorted