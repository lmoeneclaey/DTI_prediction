class Ligands:
    """
    Class the list of ligands, going to be used in the classifiers:
        - dict_ligand dict keys : DrugBankID values : smile
        - dict_ind2mol dict keys : ind values : DrugBankID
        - dict_mol2ind dict keys : DrugBankID values : ind
    """
    def __init__(self, dict_ligand, dict_ind2mol, dict_mol2ind):
        self.dict_ligand = dict_ligand
        self.dict_ind2mol = dict_ind2mol 
        self.dict_mol2ind = dict_mol2ind

class Proteins:
    """
    Class the list of ligands, going to be used in the classifiers:
        - dict_ligand dict keys : DrugBankID values : smile
        - dict_ind2mol dict keys : ind values : DrugBankID
        - dict_mol2ind dict keys : DrugBankID values : ind
    """
    def __init__(self, dict_protein, dict_ind2prot, dict_prot2ind):
        self.dict_protein = dict_protein
        self.dict_ind2prot = dict_ind2prot 
        self.dict_prot2ind = dict_prot2ind

class ListInteractions:
    """
    Class defining the interactions between a list of proteins and molecules\
        with:
            - the actual list of tuple (protein, molecule)
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

class FormattedDB:
    """
    Class defining the database, going to be used in the classifiers:

        - dict_target dict keys : UniprotID values : fasta

        - dict_intMat : matrix of interaction


        - dict_ind2_prot dict keys : ind values : UniprotID
        - dict_prot2ind dict keys : UniprotID values : ind 
    """

    def __init__(self, list_couples, interaction_bool, ind_inter):
        self.list_couples =  list_couples
        self.interaction_bool = interaction_bool
        self.ind_inter = ind_inter