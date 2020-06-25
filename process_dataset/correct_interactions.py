import copy
import numpy as np
import pandas as pd

from DTI_prediction.process_dataset.DB_utils import Drugs, Proteins, Couples, FormattedDB
from DTI_prediction.process_dataset.DB_utils import check_drug, check_protein, check_couple, get_couples_from_array

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

        print("Drug", dbid, "to orphan.")
    # dbid is a protein
    else:
        dbid_interactions_ind = np.where(couples_array[:,0]==dbid)

    print("It corresponds to", dbid_interactions_ind[0].shape[0], "interaction(s).")

    interaction_bool = DB.couples.interaction_bool
    corrected_interaction_bool = copy.deepcopy(interaction_bool)

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
    if check_couple(protein_dbid, drug_dbid, DB.couples)==True:

        couples_pd = pd.DataFrame(DB.couples.array)
        couples_pd.columns = ['UniprotID', 'DrugBankID', 'interaction_bool']
        couple_index = couples_pd[(couples_pd['UniprotID']==protein_dbid) & \
            (couples_pd['DrugBankID']==drug_dbid)].index[0]
        
        initial_interaction_bool = int(couples_pd.at[couple_index,"interaction_bool"])

        corrected_couples_pd = copy.deepcopy(couples_pd)
        if initial_interaction_bool != corrected_interaction_bool:
            corrected_couples_pd.at[couple_index, 'interaction_bool'] = corrected_interaction_bool

        corrected_couples = get_couples_from_array(couples_pd.to_numpy())

        corrected_DB = FormattedDB(drugs=DB.drugs,
                                   proteins=DB.proteins,
                                   couples=corrected_couples)



    # 2 - l'interaction n'est pas dans DB
    else:

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