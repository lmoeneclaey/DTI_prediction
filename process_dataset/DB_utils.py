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


class Interactions:
    """
    Class defining the interactions between a list of proteins and a list of \
        drugs with:
            - the actual list of tuple (protein, drug)
            - the corresponding boolean vector describing if the interaction \
                exists or not (useful to orphan interactions)
    """

    def __init__(self, couples):
        self.couples =  couples

        self.nb = self.couples.shape[0]

    def __add__(self, other):
        total_couples = np.concatenate((self.couples,
                                        other.couples),
                                        axis=0)

        return Interactions(total_couples)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

def get_intMat(drugs, proteins, interactions):
    """
    Get the matrix of interactions from a dictionary of drugs, a dictionary of \
        proteins and a list of interactions included these dictionaries.

    Parameters
    ----------
    drugs : Drugs
    proteins : Proteins
    interactions : Interactions
    """

    intMat = np.zeros((proteins.nb,
                       drugs.nb),
                       dtype=np.int32)

    for icouple in range(interactions.nb):
        protein = proteins.dict_prot2ind[interactions.couples[icouple,0]]
        drug = drugs.dict_mol2ind[interactions.couples[icouple,1]]
        interaction_bool = interactions.couples[icouple,2]
        intMat[protein, drug] = interaction_bool
    
    return intMat

class FormattedDB:
    """
    Class defining the database, going to be used in the classifiers:

        - Drugs : list of drugs
        - Proteins : list of proteins
        - ListInteractions : list of their interactions
    """

    def __init__(self, drugs, proteins, interactions):
        self.drugs =  drugs
        self.proteins = proteins
        self.interactions = interactions

        self.intMat = get_intMat(drugs, proteins, interactions)

class ListInteractions:
    """
    Class defining the interactions between a list of proteins and a list of \
        drugs with:
            - the actual list of tuple (protein, drug)
            - the corresponding boolean vector describing if the interaction \
                exists or not
            - the list of corresponding indices in the matrix of interaction
    """

    def __init__(self, list_couples, interaction_bool, ind_inter):
        self.list_couples =  list_couples
        self.interaction_bool = interaction_bool
        self.ind_inter = ind_inter

    def __add__(self, other):
        total_list_couples = self.list_couples + other.list_couples
        total_interaction_bool = np.concatenate((self.interaction_bool,
                                        other.interaction_bool),
                                        axis=0)
        total_ind_inter = (np.concatenate((self.ind_inter[0], other.ind_inter[0]), 
                                          axis=0),
                           np.concatenate((self.ind_inter[1], other.ind_inter[1]),
                                          axis=0))

        return ListInteractions(total_list_couples, 
                                total_interaction_bool,
                                total_ind_inter)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)