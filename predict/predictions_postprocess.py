import numpy as np
import pandas as pd

def predictions_postprocess_drug(predictions_output, DB, raw_proteins_df):

    predictions_df = pd.DataFrame({'UniProt ID' : list(DB.proteins.dict_protein.keys()),
                                   'prediction_moy': np.average(predictions_output, axis=1),
                                   'prediction_max': np.max(predictions_output, axis=1),
                                   'prediction_min': np.min(predictions_output, axis=1)})
    
    if (np.all(predictions_df['UniProt ID'].isin(raw_proteins_df['UniProt ID']))==False):
        print('There is a problem in the output of the prediction.')

    final_predictions_df = pd.merge(predictions_df,
                                    raw_proteins_df[['UniProt ID', 'Gene Name', 'Name']],
                                    left_on='UniProt ID',
                                    right_on='UniProt ID')

    cols = ['UniProt ID', 'Gene Name', 'Name', 'prediction_moy', 'prediction_max', 'prediction_min']

    final_predictions_df = final_predictions_df[cols]

    final_predictions_df = final_predictions_df.sort_values(by='prediction_moy', ascending=False)
    final_predictions_df = final_predictions_df.drop_duplicates()

    return final_predictions_df