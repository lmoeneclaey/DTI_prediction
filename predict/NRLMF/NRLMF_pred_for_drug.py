import argparse

from DTI_prediction.process_dataset.process_DB import get_DB
from DTI_prediction.make_kernels.get_kernels import get_K_mol_K_prot

root = './../CFTR_PROJECT/'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    "Predict the interactions "

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    parser.add_argument("--norm", default = False, action="store_true", 
                        help = "whether or not to normalize the kernels")

    parser.add_argument("--center_norm", default = False, action="store_true", 
                        help = "whether or not to center AND normalize the \
                            kernels, False by default")

    args = parser.parse_args()

    # Get the DrugBank database
    preprocessed_DB = get_DB(args.DB_version, args.DB_type)
    dict_mol2ind = preprocessed_DB.drugs.dict_mol2ind
    dict_prot2ind = preprocessed_DB.proteins.dict_prot2ind

    

    # Get the kernels of the DrugBank database
    kernels = get_K_mol_K_prot(args.DB_version, args.DB_type, args.center_norm, args.norm)
    DB_drugs_kernel = kernels[0]
    DB_proteins_kernel = kernels[1]