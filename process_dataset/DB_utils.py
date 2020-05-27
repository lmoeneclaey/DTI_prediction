import numpy as np

class Drugs:
    """
    Class defining the list of drugs, going to be used in the classifiers:
        - dict_drug dict keys : DrugBankID values : smile
        - dict_ind2mol dict keys : ind values : DrugBankID
        - dict_mol2ind dict keys : DrugBankID values : ind
    """
    def __init__(self, dict_drug, dict_ind2mol, dict_mol2ind):
        self.dict_drug = dict_drug
        self.dict_ind2mol = dict_ind2mol 
        self.dict_mol2ind = dict_mol2ind

        self.nb = len(list(self.dict_drug.keys()))

class Proteins:
    """
    Class defining the list of proteins, going to be used in the classifiers:
        - dict_protein dict keys : UniprotID values : fasta
        - dict_ind2_prot dict keys : ind values : UniprotID
        - dict_prot2ind dict keys : UniprotID values : ind  
    """
    def __init__(self, dict_protein, dict_ind2prot, dict_prot2ind):
        self.dict_protein = dict_protein
        self.dict_ind2prot = dict_ind2prot 
        self.dict_prot2ind = dict_prot2ind

        self.nb = len(list(self.dict_protein.keys()))


class Couples:
    """
    Class defining couples between a list of proteins and a list of \
        drugs with:
            - the actual list of tuple (protein, drug)
            - the corresponding boolean vector describing if the interaction \
                exists or not (useful to orphan interactions)
    """

    def __init__(self, list_couples, interaction_bool):

        self.list_couples = list_couples
        self.interaction_bool = interaction_bool


        self.array = np.concatenate((np.array(self.list_couples), 
                                     self.interaction_bool),
                                     axis = 1)
        self.nb = len(self.list_couples)

    def __add__(self, other):

        total_list_couples = self.list_couples + other.list_couples
        total_interaction_bool = np.concatenate((self.interaction_bool,
                                                other.interaction_bool),
                                                axis=0) 

        return Couples(total_list_couples, total_interaction_bool)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

def get_intMat(drugs, proteins, couples):
    """
    Get the matrix of interactions from a dictionary of drugs, a dictionary of \
        proteins and a list of couples included these dictionaries.

    Parameters
    ----------
    drugs : Drugs
    proteins : Proteins
    couples : Couples
    """

    intMat = np.zeros((proteins.nb,
                       drugs.nb),
                       dtype=np.int32)

    for icouple in range(couples.nb):
        protein = proteins.dict_prot2ind[couples.array[icouple,0]]
        drug = drugs.dict_mol2ind[couples.array[icouple,1]]
        interaction_bool = couples.array[icouple,2]
        intMat[protein, drug] = interaction_bool
    
    return intMat

def get_interactions(couples):
    """
    From a 'Couples' object of a 'FormattedDB', get only the "true" interactions
        corresponding to 1.

    Parameters
    ----------
    couples : Couples
    """

    interactions_ind = np.where(couples.array[:,2]=='1')
    interactions_arr = couples.array[interactions_ind]

    list_interactions = interactions_arr[:,:2].tolist()
    interactions = Couples(list_couples = list_interactions,
                           interaction_bool = couples.interaction_bool[interactions_ind])

    if np.all(interactions.interaction_bool == 1)==False:
        print("There is an error in the function get_interactions()")

    return interactions 

class FormattedDB:
    """
    Class defining the database, going to be used in the classifiers:

        - drugs : Drugs
        - proteins : Proteins
        - couples : Couples
    """

    def __init__(self, drugs, proteins, couples):
        self.drugs =  drugs
        self.proteins = proteins
        self.couples = couples

        self.interactions = get_interactions(self.couples)
        self.intMat = get_intMat(drugs, proteins, couples)