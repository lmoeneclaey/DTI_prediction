import numpy as np

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, check_protein, check_couple

def get_orphan(DB, dbid):
    """
    Correct 1 to 0 in the matrix of interactions, all the interactions concerning\
        one molecule or one protein

    Parameters
    ----------
    DB : FormattedDB
        got with the function process_dataset.process_DB.get_DB()
    dbid : str

    Returns
    -------
    corrected_DB : FormattedDB
    """

    couples_array = DB.couples.array

    # dbid is a drug
    if dbid[:2] == 'DB':
        dbid_interactions_ind = np.where(couples_array[:,1]==dbid)
    # dbid is a protein
    else:
        dbid_interactions_ind = np.where(couples_array[:,0]==dbid)

    interaction_bool = DB.couples.interaction_bool
    corrected_interaction_bool = interaction_bool

    for ind in dbid_interactions_ind[0]:
        if interaction_bool[ind]==1:
            corrected_interaction_bool[ind]=0

    corrected_couples = Couples(list_couples=DB.couples.list_couples,
                                interaction_bool=corrected_interaction_bool)

    corrected_DB = FormattedDB(drugs=DB.drugs,
                               proteins=DB.proteins,
                               couples=corrected_couples)

    return corrected_DB


# Maybe later change to have smaller functions
def correct_interactions(protein_dbid, drug_dbid, corrected_interaction_bool, DB):
    """
    Correct 1 to 0 in the matrix of interactions, interactions that haven't \
    been proven experimentally.

    Parameters
    ----------
    interaction : tuple of length 2
        (UniprotID, DrugbankID)
    DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    corrected_DB : tuple of length 8 
    """

    # 1 - l'interaction est déjà dans DB 
    # (cela veut dire que drug_dbid est dans Drugs et protein_dbid est dans Proteins)
    # on va l'admettre mais il faudrait le vérifier à un moment

    # 2 - l'interaction n'est pas dans DB
    if check_couple(protein_dbid, drug_dbid, DB.couples)==False:

    # 2A - drug_dbid est dans Drugs, protein_dbid est dans Proteins
        if check_protein(protein_dbid, DB.proteins):

            if check_drug(drug_dbid, DB.drugs):

                new_couple = Couples(list_couples=[(protein_dbid, drug_dbid)],
                                     interaction_bool=np.array([corrected_interaction_bool]).reshape(-1,1))

                corrected_couples = DB.couples + new_couple

    # 2B - drug_dbid n'est pas dans Drugs 

    # 2C - protein_dbid n'est pas dans Proteins

    corrected_DB = FormattedDB(drugs=DB.drugs,
                               proteins=DB.proteins,
                               couples=corrected_couples)

    return corrected_DB