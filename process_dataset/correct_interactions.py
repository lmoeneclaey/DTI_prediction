import numpy as np

from process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB

def get_orphan(dbid, preprocessed_DB):
    """
    Correct 1 to 0 in the matrix of interactions, all the interactions concerning\
        one molecule or one protein

    Parameters
    ----------
    dbid : str
    preprocessed_DB : FormattedDB
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    corrected_DB : FormattedDB
    """

    couples_array = preprocessed_DB.couples.array

    # dbid is a drug
    if dbid[:2] == 'DB':
        dbid_interactions_ind = np.where(couples_array[:,1]==dbid)
    # dbid is a protein
    else:
        dbid_interactions_ind = np.where(couples_array[:,0]==dbid)

    interaction_bool = preprocessed_DB.couples.interaction_bool
    corrected_interaction_bool = interaction_bool

    for ind in dbid_interactions_ind[0]:
        if interaction_bool[ind]==1:
            corrected_interaction_bool[ind]=0

    corrected_couples = Couples(list_couples=preprocessed_DB.couples.list_couples,
                                interaction_bool=corrected_interaction_bool)

    corrected_DB = FormattedDB(drugs=preprocessed_DB.drugs,
                               proteins=preprocessed_DB.proteins,
                               couples=corrected_couples)

    return corrected_DB


def correct_unproven_interactions(interaction, preprocessed_DB):
    """
    Correct 1 to 0 in the matrix of interactions, interactions that haven't \
    been proven experimentally.

    Parameters
    ----------
    interaction : tuple of length 2
        (UniprotID, DrugbankID)
    preprocessed_DB : tuple of length 8
        got with the function process_dataset.process_DB.get_DB()

    Returns
    -------
    corrected_preprocessed_DB : tuple of length 8 
    """