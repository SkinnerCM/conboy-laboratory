import pandas as pd 
import numpy as np 
from scipy import stats

def prep_dataset(data):
    """
    Prepare a dataset for calculating age correlations.

    This function expects a DataFrame with two columns:
    - The first column should contain the CpG sites for the model.
    - The second column should contain the corresponding weights or importance scores.

    The function renames the columns to 'CpG' and 'Weight' and sorts the DataFrame
    by the 'CpG' column.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
    pd.DataFrame: The processed DataFrame with columns renamed and sorted by 'CpG'.
    """
    # Ensure the data has the correct column names
    data.columns = ['CpG', 'Weight']
    
    # Sort the data by CpG sites
    data = data.sort_values('CpG')
    
    return data

def get_preds(inputs, model):
    
    """
    Generate predictions for a given model using the provided input data.

    This function checks if the input data contains an intercept column; if not, 
    it adds one. It then verifies that the features in the input data match the 
    features expected by the model. If they do, it computes predictions by taking 
    the dot product of the model weights and the input features for each row of data.

    Parameters:
    -----------
    inputs : pandas.DataFrame
        A DataFrame containing the input data. Each row represents a sample, and 
        each column represents a feature. If the first column is not named 'Intercept', 
        the function will add a column of ones as the intercept.
        
    model : object
        A model object that has the following attributes:
        - `CpG`: a list of feature names that the model expects (including 'Intercept').
        - `Weight`: a numpy array or pandas Series containing the model weights, 
          where the order corresponds to the features listed in `CpG`.

    Returns:
    --------
    list
        A list of predictions, one for each row in the input data.

    Raises:
    -------
    ValueError
        If the features in the input data do not match the features expected by the model.
    """
    
    if inputs.columns[0]!='Intercept':
        # Insert a column of ones at the front
        inputs.insert(0, 'Intercept', 1)
    
    if inputs.columns.tolist()[1:]==model.CpG.tolist()[1:]:
        
        preds = []

        for row in inputs.index.tolist():
            # Calculate the dot product
            preds+= [model.Weight.dot(inputs.loc[row].values)]
        return preds
        
    else:
        return print('error: feature mismatch')

