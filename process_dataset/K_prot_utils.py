import numpy as np
import pickle
import sys
import math
import os
import argparse

from sklearn.preprocessing import KernelCenterer

from process_DB import get_DB
from make_K_mol import center_and_normalise_kernel
from make_K_prot import make_temp_Kprot

root = './../CFTR_PROJECT/'

def make_range_temp_Kprot(DB_version, DB_type, process_name, i1, i2):
    """ 
    Process make_temp_Kprot() for a range of proteins

    The proteins got keys between *i1* and *i2* in the dict_target dictionary
    See the description of make_temp_Kprot for more details 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'
    i1 : int
        index of the first protein in the range in the dictionaries
    i2 : int
        index of the last protein in the range in the dictionaries

    Returns
    -------
    None
    """

    for index in range(i1, i2):
        print(index)
        make_temp_Kprot(DB_version, DB_type, process_name, index)


def check_temp_Kprot(DB_version, DB_type, process_name):
    """ 
    Check and process make_temp_Kprot() for the proteins for which the \
    LAkernel has not been processed.

    See the description of make_temp_Kprot for more details 

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'

    Returns
    -------
    None
    """

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[3]
    dict_ind2prot = preprocessed_DB[4]

    nb_prot = len(list(dict_target.keys()))

    list_ = []
    for index in range(nb_prot):

        # output_filename
        dbid = dict_ind2prot[index]
        output_filename = root + data_dir + 'LAkernel/LA_' + pattern_name + \
            '_' + dbid + '.txt'

        if not os.path.isfile(output_filename):
            list_.append(dbid)
    print("list of uncompleted proteins", list_)

#    for index in list_:
#       make_temp_Kprot(DB_version, DB_type, process_name, index)

def del_temp_Kprot(DB_version, DB_type, process_name, delete):
    """ 
    Check (and -optional- delete) LAkernel output files which are not completed.  

    Parameters
    ----------
    DB_version : str
        string of the DrugBank version number
        format : "drugbank_vX.X.X" exemple : "drugbank_v5.1.1"
    DB_type : str
        string of the DrugBank type exemple: "S0h"
    process_name : str
        string of the process name exemple: 'NNdti'
    delete : boolean
        whether or not to delete the file 

    Returns
    -------
    None
    """

    # pattern_name variable
    pattern_name = process_name + '_' + DB_type
    # data_dir variable 
    data_dir = 'data/' + DB_version + '/' + pattern_name + '/'

    # get the DBdataBase preprocessed
    preprocessed_DB = get_DB(DB_version, DB_type, process_name)
    dict_target = preprocessed_DB[3]
    dict_ind2prot = preprocessed_DB[4]

    nb_prot = len(list(dict_target.keys()))

    list_ = []
    for index in range(nb_prot):

        # output_filename
        dbid = dict_ind2prot[index]
        output_filename = root + data_dir + 'LAkernel/LA_' + pattern_name + \
            '_' + dbid + '.txt'

        if os.path.isfile(output_filename):
            output_file = open(output_filename)
            count_nb_line = 0
            for line in output_file.readlines(): count_nb_line += 1
            if count_nb_line != nb_prot - index:
                list_.append(output_filename)
        else:
            list_.append(output_filename)
    print(list_)

    print('delete', delete)

    if delete==True:
        for outf in list_:
            if os.path.isfile(outf):
                os.remove(outf)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    "Other functions, helpful to survey the LA Kernel process")

    # we want to choose between different string for action
    parser.add_argument("action", type = str, choices = ["temp_range", "check",\
        "del"], help = "if you want to :\
            - process the similarity for a range of proteins\
            - check uncompleted kernels\
            - or delete uncompleted kernels")

    parser.add_argument("DB_version", type = str, choices = ["drugbank_v5.1.1",
                        "drugbank_v5.1.5"], help = "the number of the DrugBank \
                            version, example: 'drugbank_vX.X.X'")

    # to change
    parser.add_argument("DB_type", type = str,
                        help = "the DrugBank type, example: 'S0h'")

    parser.add_argument("process_name", type = str,
                        help = "the name of the process, helper to find the \
                        data again, example = 'DTI'")

    # need to change for the uniprotID
    parser.add_argument("-i1", "--index1", type = int,
                        help = "the range of the protein in question, \
                        exclusively for make_range_temp_Kprot()")
    
    parser.add_argument("-i2", "--index2", type = int,
                        help = "the range of the protein in question, \
                        exclusively for make_range_temp_Kprot()")

    parser.add_argument("--delete", action="store_true",
                        help = "whether or not to delete the file, exclusively \
                        for del_temp_Kprot()")

    # parser.add_argument("-v", "--verbosity", action = "store_true", 
                        # help = "increase output verbosity")

    args = parser.parse_args()

    if args.action == "temp_range":
        make_range_temp_Kprot(args.DB_version, args.DB_type, args.process_name,\
            args.index1, args.index2)

    elif args.action == "check":
        check_temp_Kprot(args.DB_version, args.DB_type, args.process_name)
    
    elif args.action == "del":
        del_temp_Kprot(args.DB_version, args.DB_type, args.process_name, \
            args.delete)
