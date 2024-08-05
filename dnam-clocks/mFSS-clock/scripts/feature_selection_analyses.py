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

def get_stats(dataset, meta):
    """
    Compute regression statistics for each column in the dataset.

    This function performs linear regression for each column in the dataset
    against the age data provided in the meta DataFrame. It calculates the
    correlation coefficient and standard error for each regression.

    Parameters:
    dataset (pd.DataFrame): A DataFrame where each column contains CpG site data
                            for regression analysis.
    meta (pd.DataFrame): A DataFrame containing metadata with an 'age' column containing sample ages.

    Returns:
    tuple: A tuple containing two lists:
        - rs (list of float): The correlation coefficients (r-values) from the regression.
        - stderrs (list of float): The standard errors of the regression slopes.
    """
    rs = []
    stderrs = []

    for column in dataset:
        # Perform linear regression between age and the current column
        regression = stats.linregress(meta.age.astype(float), dataset[column].astype(float))
        slope, intercept, rvalue, pvalue, stderr = regression
        rs.append(rvalue)
        stderrs.append(stderr)
        
    return rs, stderrs

def get_tvals(weights, stderrs):
    """
    Calculate feature importances based on t-values.

    This function computes the t-values for each feature using the formula:
    t = weight / standard error. The t-values indicate the significance of each
    feature's weight.

    Parameters:
    weights (list of float): A list of feature weights.
    stderrs (list of float): A list of standard errors corresponding to each feature weight.

    Returns:
    list of float: A list of t-values for each feature.
    """
    tvals = []
    for i in range(len(weights)):
        tvals.append(weights[i] / stderrs[i])
        
    return tvals

def model_corrs(model, ref_data, meta):
    """
    Calculate the correlations and feature importances between a model and reference dataset.

    This function identifies the intersection of CpG sites between a reference dataset and a model dataset,
    computes the age-correlations and t-statistics for the CpGs in the model, and returns a DataFrame with
    the results. It also identifies any CpG sites present in the model but missing from the reference dataset.

    Parameters:
    model (pd.DataFrame): DataFrame containing CpGs and their associated weights for the given model.
                         The DataFrame must have columns 'CpG' and 'Weight'.
    ref_data (pd.DataFrame): DataFrame containing the reference dataset with sample beta values.
    meta (pd.DataFrame): DataFrame containing metadata with an 'age' column containing sample ages for computing correlations.

    Returns:
    pd.DataFrame: DataFrame containing the model CpGs, model weights, age-correlations (r), t-statistics (t),
                  and R-squared values (R2) for each CpG.
    list: List of CpG sites present in the model but missing from the reference dataset.
    """

    # Get the intersection of the CpGs between the model and the reference data
    intersect = list(set(ref_data.columns) & set(model.CpG))
    
    # Identify missing CpGs and sort them
    missing = sorted(list(set(model.CpG) - set(intersect)))
    
    # Sort the intersection to keep the lists synced
    intersect.sort()
    
    # Filter the model data to only include CpGs in the intersection
    data = model.loc[model.CpG.isin(intersect)]
    
    # Extract the CpG values from the reference data for the intersection
    combined = ref_data[intersect]
    
    # Compute age-correlations and standard errors
    model_rs, stderrs = get_stats(combined, meta)
    
    # Compute the feature importances based on t-statistics
    importances = get_tvals(model.Weight.tolist(), stderrs)
    
    # Create a DataFrame with the results
    cg_corrs = pd.DataFrame({
        'CpG': combined.columns.tolist(),
        'Weight': model.Weight.tolist(),
        'r': model_rs,
        't': importances
    })
    cg_corrs['R2'] = cg_corrs.r ** 2 
    
    return cg_corrs, missing

def normalized_importance(data):

    """
    This function calculates the normalized importances for the feature selection of a model.

    Parameters:
    data (pd.DataFrame): The output dataframe from the model_corrs function, containing a column "t" with t-statistics for features.

    Returns:
    pd.Series: Series of normalized feature importances, with the same index as the input DataFrame.

    """
    
    importances = abs(data.t)/abs(data.t).max()
    
    return importances


def het_r(data, model, ages):
    """
    Measure the noise (heteroscedasticity) with age for a given predictor by calculating the residuals 
    for the age-correlation and then regressing those residuals on age.

    Parameters:
    data (pd.DataFrame): Processed beta values for the CpG selection of a given model (e.g., Hannum, PhenoAge, DunedinPACE, etc.).
    model (pd.DataFrame): DataFrame containing CpGs and their associated weights for the given model.
                         The DataFrame must have columns 'CpG' and 'Weight'.
    ages (list): List of sample ages corresponding to the beta values in `data`.

    Returns:
    list: List of R-squared values for the correlation between age and noise for each CpG in the given model.
    """
    
    # Get the intersection of the CpGs between the model and the data
    intersect = list(set(data.columns) & set(model.CpG))
    
    # Sort the intersection to keep the lists synced
    intersect.sort()
    
    temp_data = data[intersect]
    het_rs = []
    
    for cg in temp_data.columns:
        # Regress a given predictor on age
        regression = stats.linregress(ages.astype(float), temp_data[cg].tolist())
        slope, intercept, rvalue, pvalue, stderr = regression
        
        preds = ages * slope + intercept
        actuals = temp_data[cg].tolist()
        
        # Calculate the absolute residuals
        abs_resids = abs(preds - actuals)
        
        # Regress the residuals for the predictor on age
        regression = stats.linregress(ages.astype(float), abs_resids)
        slope, intercept, rvalue, pvalue, stderr = regression
        
        het_rs.append(rvalue * rvalue)
        
    return het_rs


