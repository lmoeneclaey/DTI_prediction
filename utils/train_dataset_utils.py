import numpy as np
import pandas as pd

from DTI_prediction.process_dataset.process_DB import get_DB

root = './../CFTR_PROJECT/'

def get_nb_interactions_per_prot(list_train_datasets, DB_version, DB_type):

    DB = get_DB(DB_version, DB_type)

    nb_clf = len(list_train_datasets) 

    mean_nb_positive_interactions_per_prot_list = []
    mean_nb_negative_interactions_per_prot_list = []

    for prot_id in list(DB.proteins.dict_ind2prot.values()):
    
        nb_positive_interactions_per_prot_list = []
        nb_negative_interactions_per_prot_list = []

        for iclf in range(nb_clf):
            train_dataset = pd.DataFrame(list_train_datasets[iclf].array, columns=['UniProt ID', 'DrugbankID', 'interaction_bool'])

            positive_interactions = train_dataset[train_dataset['interaction_bool']=='1']
            negative_interactions = train_dataset[train_dataset['interaction_bool']=='0']

            prot_positive_interactions = positive_interactions[positive_interactions['UniProt ID']==prot_id]
            prot_negative_interactions = negative_interactions[negative_interactions['UniProt ID']==prot_id]

            nb_positive_interactions_per_prot_list.append(prot_positive_interactions.shape[0])
            nb_negative_interactions_per_prot_list.append(prot_negative_interactions.shape[0])

        mean_nb_positive_interactions_per_prot_list.append(np.average(nb_positive_interactions_per_prot_list))
        mean_nb_negative_interactions_per_prot_list.append(np.average(nb_negative_interactions_per_prot_list))

    nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(DB.proteins.dict_ind2prot.values()),
                                            'Nb_positive_interactions':mean_nb_positive_interactions_per_prot_list,
                                            'Nb_negative_interactions':mean_nb_negative_interactions_per_prot_list})

    nb_interactions_per_prot = nb_interactions_per_prot.sort_values(by=['Nb_positive_interactions'], ascending=False)

    raw_data_dir = 'data/' + DB_version + '/raw/'
    raw_df = pd.read_csv(root + raw_data_dir + \
                    'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                    sep=",")
    raw_df = raw_df.fillna('')

    final_nb_interactions = pd.merge(nb_interactions_per_prot,
                                raw_df[['UniProt ID', 'Gene Name', 'Name']],
                                left_on='UniProt ID',
                                right_on='UniProt ID')

    final_nb_interactions = final_nb_interactions.drop_duplicates()
    
    balanced_bool = np.all(final_nb_interactions['Nb_positive_interactions'\
        ]==final_nb_interactions['Nb_negative_interactions'])

    if balanced_bool:
        print("The classfifiers are balanced.")
    else:
        print("Warning: The classifiers are not balanced in terms of number of\
            positive and negative interactions per protein.")
    
    return final_nb_interactions

def is_orphan(list_train_datasets, dbid):

    nb_clf =len(list_train_datasets)

    list_bool_orphan = []

    for iclf in range(nb_clf):

        train_dataset = pd.DataFrame(list_train_datasets[iclf].array, columns=['UniProt ID', 
                                                                               'DrugbankID', 
                                                                               'interaction_bool'])
        if dbid[:2] == 'DB':
            dbid_interactions = train_dataset[train_dataset['DrugbankID']==dbid]
        else:
            dbid_interactions = train_dataset[train_dataset['UniProt ID']==dbid]

        if dbid_interactions.shape[0]>0:
            list_bool_orphan.append(np.all(dbid_interactions['interaction_bool']=='0'))
        else:
            list_bool_orphan.append(True)

    if np.all(list_bool_orphan):
        print(dbid, "is orphan in all the train datasets.")
        print(list_bool_orphan)
    else: 
        print("Warning:", dbid, " is not orphan in the train datasets")
        print(list_bool_orphan)