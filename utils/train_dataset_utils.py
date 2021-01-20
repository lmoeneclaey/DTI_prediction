import numpy as np
import pandas as pd

from DTI_prediction.process_dataset.process_DB import get_DB

root = './../CFTR_PROJECT/'

# def get_nb_interactions_per_prot(list_train_datasets, DB_version, DB_type):

#     DB = get_DB(DB_version, DB_type)

#     nb_clf = len(list_train_datasets) 

#     mean_nb_positive_interactions_per_prot_list = []
#     mean_nb_negative_interactions_per_prot_list = []

#     for prot_id in list(DB.proteins.dict_ind2prot.values()):
    
#         nb_positive_interactions_per_prot_list = []
#         nb_negative_interactions_per_prot_list = []

#         for iclf in range(nb_clf):
#             train_dataset = pd.DataFrame(list_train_datasets[iclf].array, columns=['UniProt ID', 'DrugbankID', 'interaction_bool'])

#             positive_interactions = train_dataset[train_dataset['interaction_bool']=='1']
#             negative_interactions = train_dataset[train_dataset['interaction_bool']=='0']

#             prot_positive_interactions = positive_interactions[positive_interactions['UniProt ID']==prot_id]
#             prot_negative_interactions = negative_interactions[negative_interactions['UniProt ID']==prot_id]

#             nb_positive_interactions_per_prot_list.append(prot_positive_interactions.shape[0])
#             nb_negative_interactions_per_prot_list.append(prot_negative_interactions.shape[0])

#         mean_nb_positive_interactions_per_prot_list.append(np.average(nb_positive_interactions_per_prot_list))
#         mean_nb_negative_interactions_per_prot_list.append(np.average(nb_negative_interactions_per_prot_list))

#     nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(DB.proteins.dict_ind2prot.values()),
#                                             'Nb_positive_interactions':mean_nb_positive_interactions_per_prot_list,
#                                             'Nb_negative_interactions':mean_nb_negative_interactions_per_prot_list})

#     nb_interactions_per_prot = nb_interactions_per_prot.sort_values(by=['Nb_positive_interactions'], ascending=False)

#     raw_data_dir = 'data/' + DB_version + '/raw/'
#     raw_df = pd.read_csv(root + raw_data_dir + \
#                     'drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
#                     sep=",")
#     raw_df = raw_df.fillna('')

#     final_nb_interactions = pd.merge(nb_interactions_per_prot,
#                                 raw_df[['UniProt ID', 'Gene Name', 'Name']],
#                                 left_on='UniProt ID',
#                                 right_on='UniProt ID')

#     final_nb_interactions = final_nb_interactions.drop_duplicates()
    
#     balanced_bool = np.all(final_nb_interactions['Nb_positive_interactions'\
#         ]==final_nb_interactions['Nb_negative_interactions'])

#     if balanced_bool:
#         print("The classfifiers are balanced.")
#     else:
#         print("Warning: The classifiers are not balanced in terms of number of\
#             positive and negative interactions per protein.")
    
#     return final_nb_interactions

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

def get_number_of_interactions_per_mol(train_dataset_pd, test_dataset_pd):

    test_mol_involved = np.unique(test_dataset_pd['DrugBank ID'])

    test_nb_interactions_per_mol_list = []
    for mol_id in list(test_mol_involved):
        test_inner_mol_interactions = train_dataset_pd[(train_dataset_pd['DrugBank ID']==mol_id) &
                                                    (train_dataset_pd['interaction_bool']==1)]
        test_nb_interactions_per_mol_list.append(test_inner_mol_interactions.shape[0])

    test_nb_interactions_per_mol = pd.DataFrame({'DrugBank ID':list(test_mol_involved),
                                                   'Nb_interactions_per_drug':test_nb_interactions_per_mol_list})

    # Adapt the number of categories
    category = []
    for val in test_nb_interactions_per_mol['Nb_interactions_per_drug']:
        if val==0:
            category.append('0')
        elif 1 <= val <=4:
            category.append('[1,4]')
        elif 5 <= val <= 10:
            category.append('[5,10]')
        else:
            category.append('> 10')

    test_nb_interactions_per_mol['category_drug']=category

    result = pd.merge(test_dataset_pd, test_nb_interactions_per_mol[['DrugBank ID', 'category_drug']], on='DrugBank ID', how="left")

    return result

def get_number_of_interactions_per_prot(train_dataset_pd, test_dataset_pd):

    test_prot_involved = np.unique(test_dataset_pd['UniProt ID'])

    test_nb_interactions_per_prot_list = []
    for prot_id in list(test_prot_involved):
        test_inner_prot_interactions = train_dataset_pd[(train_dataset_pd['UniProt ID']==prot_id) &
                                                    (train_dataset_pd['interaction_bool']==1)]
        test_nb_interactions_per_prot_list.append(test_inner_prot_interactions.shape[0])

    test_nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(test_prot_involved),
                                                 'Nb_interactions_per_prot':test_nb_interactions_per_prot_list})

    result = pd.merge(test_dataset_pd, test_nb_interactions_per_prot[['UniProt ID', 'Nb_interactions_per_prot']], on='UniProt ID', how="left")

    return result

def get_number_of_interactions_cat_per_prot(train_dataset_pd, test_dataset_pd):

    test_prot_involved = np.unique(test_dataset_pd['UniProt ID'])

    test_nb_interactions_per_prot_list = []
    for prot_id in list(test_prot_involved):
        test_inner_prot_interactions = train_dataset_pd[(train_dataset_pd['UniProt ID']==prot_id) &
                                                    (train_dataset_pd['interaction_bool']==1)]
        test_nb_interactions_per_prot_list.append(test_inner_prot_interactions.shape[0])

    test_nb_interactions_per_prot = pd.DataFrame({'UniProt ID':list(test_prot_involved),
                                                 'Nb_interactions_per_prot':test_nb_interactions_per_prot_list})

    # Adapt the number of categories
    category = []
    # for val in test_nb_interactions_per_prot['Nb_interactions']:
    #     if val==0:
    #         category.append('0')
    #     elif 1 <= val <=4:
    #         category.append('[1,4]')
    #     elif 5 <= val <= 10:
    #         category.append('[5,10]')
    #     else:
    #         category.append('> 10')

    for val in test_nb_interactions_per_prot['Nb_interactions_per_prot']:
        if val==0:
            category.append('0')
        elif val==1:
            category.append('1')
        elif 2 <= val <=4:
            category.append('[2,4]')
        elif 5 <= val <= 10:
            category.append('[5,10]')
        elif 11 <= val <= 20:
            category.append('[11,20]')
        elif 21 <= val <= 30:
            category.append('[21,30]')
        else:
            category.append('> 30')


    test_nb_interactions_per_prot['category_prot']=category

    result = pd.merge(test_dataset_pd, test_nb_interactions_per_prot[['UniProt ID', 'category_prot']], on='UniProt ID', how="left")

    return result