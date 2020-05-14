from rdkit import Chem

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