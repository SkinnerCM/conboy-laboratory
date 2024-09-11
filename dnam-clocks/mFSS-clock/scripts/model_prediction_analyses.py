import pandas as pd 
import numpy as np 
from scipy import stats
from utils.data_processing import prep_model


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
    
def get_horvath_preds(inputs, model):
    
    """
    Compute Horvath clock predictions based on input features and model weights.

    This function calculates predictions after applying to the inputs the inverse log transform
    used by the Horvath clock model. It first checks if the inputs have an 'Intercept' column, adding one if not. 
    Then, it verifies that the CpG features in the input match those in the model 
    before calculating predictions using a dot product of the model's weights 
    and the transformed input feature values.

    Parameters:
    -----------
    inputs : pandas.DataFrame
        A DataFrame of input features for prediction. The first column should be 
        an 'Intercept' column (a column of ones), and the remaining columns should 
        match the CpG sites expected by the model.
        
    model : pandas.DataFrame
        A DataFrame representing the Horvath clock model with 'Weight' and 'CpG' 
        columns. 'Weight' is used for the dot product calculation with the inputs, 
        and 'CpG' should match the feature names in the input data.

    Returns:
    --------
    preds : list
        A list of Horvath clock predictions for each input row.
    
    Raises:
    -------
    ValueError:
        If there is a feature mismatch between the input data and the model's CpG sites.
    """
    
    if inputs.columns[0]!='Intercept':
        # Insert a column of ones at the front
        inputs.insert(0, 'Intercept', 1)
    
    if inputs.columns.tolist()[1:]==model.CpG.tolist()[1:]:
        
        preds = []

        for row in inputs.index.tolist():
            # Calculate the dot product
            out = model.Weight.dot(inputs.loc[row].values)
            
            if out <= 0:
                
                preds+=[21*10**out -1]
                
            else:
                
                preds+=[21*out+20]
                
        return preds
        
    else:
        return print('error: feature mismatch')

    
def get_residuals(preds, metadata, model_name):
    
    """
    Calculate residuals for a given model's predictions on a dataset.

    This function adds model predictions to the metadata and computes the 
    residuals by subtracting the actual age from the predicted values. 
    It is designed for use with a specific cohort from the V7 combined 
    composite dataset (GSE42861, GSE125105, GSE72774, GSE106648).

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data used for making predictions with the model.
        
    metadata : pandas.DataFrame
        Metadata containing the true age values for the dataset, and which 
        will also be used to store the model's predictions under `model_name`.
        
    model : sklearn-like model
        The trained machine learning model used to generate predictions.
        
    model_name : str
        A string indicating the column name in the metadata where the model's 
        predictions will be stored.

    Returns:
    --------
    residuals : pandas.Series
        A series of residuals calculated as the difference between the model's 
        predictions and the actual age values from the metadata.
    """
    
    #get the model predictions for the dataset and add them to the relevant metadata
    metadata.loc[:, model_name] = preds
    
    
    residuals = metadata[model_name]-metadata.age
    
    return residuals

def get_mu_std(residuals):
    
    """
    Fit a normal distribution to the residuals and return the mean (mu) and 
    standard deviation (std).

    This function fits a normal distribution to the provided residuals using 
    maximum likelihood estimation and returns the estimated mean and standard 
    deviation.

    Parameters:
    -----------
    residuals : array-like
        A series or array of residuals to which a normal distribution is fitted.

    Returns:
    --------
    mu : float
        The estimated mean of the fitted normal distribution.

    std : float
        The estimated standard deviation of the fitted normal distribution.
    """
    
    # Fit normal distribution parameters
    params_norm = stats.norm.fit(residuals)
    mu, std = params_norm
    return mu, std

def get_cutoffs(mu, std, alpha=0.05):
    
    """
    Calculate the lower and upper cutoffs for a normal distribution given 
    the mean (mu), standard deviation (std), and significance level (alpha).

    This function calculates the cutoff values that correspond to the lower 
    and upper `alpha/2` percentiles of a normal distribution with a specified 
    mean and standard deviation. These cutoffs represent the bounds for a 
    two-tailed test.

    Parameters:
    -----------
    mu : float
        The mean of the normal distribution.
        
    std : float
        The standard deviation of the normal distribution.
        
    alpha : float, optional
        The significance level for the two-tailed test. Default is 0.05, 
        representing a 95% confidence interval (cutoffs at 2.5% and 97.5%).

    Returns:
    --------
    lower_cutoff : float
        The lower cutoff value (alpha/2 percentile).
    
    upper_cutoff : float
        The upper cutoff value (1 - alpha/2 percentile).
    """
     
    lower_cutoff = round(stats.norm.ppf(alpha/2, loc=mu, scale=std),1)
    upper_cutoff = round(stats.norm.ppf(1 - alpha/2, loc=mu, scale=std),1)
    
    return lower_cutoff, upper_cutoff

def create_residual_distribution(residuals, mu, std, model_name):
    
    """
    Create a probability density function (PDF) for the residuals based on 
    a fitted normal distribution and return the distribution as a DataFrame.

    This function generates a range of values from the minimum to the maximum 
    of the residuals, computes the PDF using the provided mean (mu) and 
    standard deviation (std), and returns a DataFrame containing the error 
    values, the density of the distribution, and the model name.

    Parameters:
    -----------
    residuals : array-like
        A series or array of residuals used to define the range for the PDF.
        
    mu : float
        The mean of the fitted normal distribution.
        
    std : float
        The standard deviation of the fitted normal distribution.
        
    model_name : str
        The name of the model, used for labeling the distribution.

    Returns:
    --------
    model_fit : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Error': The range of error values (x-axis) for the distribution.
        - 'Density': The corresponding density values (y-axis) from the PDF.
        - 'Model': The name of the model associated with the distribution.
    """
    
    # Create PDFs (probability density functions) of the fitted distributions
    xmin = residuals.min()
    xmax = residuals.max()
    model_x = np.linspace(xmin, xmax, 1000)
    model_resids = stats.norm.pdf(model_x, mu, std)

    model_fit = pd.DataFrame({'Error': model_x, 'Density': model_resids, 'Model': model_name})
    
    return model_fit

def model_errs_and_dist(data, metadata, model, model_name, horvath_model=False):
    
    """
    Calculate residuals, fit a normal distribution, and generate a residual distribution 
    for a given model, returning key statistics and the distribution.

    This function computes the residuals from the model predictions, fits a normal 
    distribution to the residuals, calculates the cutoffs for the distribution, and 
    creates the residual distribution. It returns a list of statistics and a DataFrame 
    of the residual distribution for the model.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data used for making predictions with the model.
    
    metadata : pandas.DataFrame
        The metadata containing true age values and used to store the model predictions.
    
    model : sklearn-like model
        The machine learning model used to generate predictions.
    
    model_name : str
        The name of the model, used for labeling statistics and the distribution.
        
    horvath_model : bool
        Whether or not the model is Horvath

    Returns:
    --------
    model_stats : list of tuples
        A list containing a tuple with the model name, lower cutoff, upper cutoff, 
        and the mean (mu) of the residuals rounded to 2 decimal places.

    model_fit : pandas.DataFrame
        A DataFrame containing the error values, density, and model name for the 
        residual distribution.
    """
    
    model = prep_model(model)
    
    if horvath_model:
        preds = get_horvath_preds(data, model)
        
    else:
        preds = get_preds(data, model)
        
    residuals = get_residuals(preds,metadata, model_name)
    mu, std = get_mu_std(residuals)
    cutoffs = get_cutoffs(mu, std)
    model_fit = create_residual_distribution(residuals, mu, std, model_name)
    
    model_stats = [(model_name,cutoffs[0], cutoffs[1], round(mu, 2))]
    
    return model_stats, model_fit

def cohens_d(s1, s2):
    
    """
    Calculate Cohen's d, a measure of effect size, between two independent samples.

    This function computes Cohen's d by calculating the difference between the means 
    of two samples (s1 and s2), and dividing by the pooled standard deviation. 
    Cohen's d is commonly used to indicate the standardized difference between two means.

    Parameters:
    -----------
    s1 : array-like
        The first sample of data (e.g., control group).
    
    s2 : array-like
        The second sample of data (e.g., treatment group).

    Returns:
    --------
    d : float
        The computed Cohen's d value, representing the effect size.

    Notes:
    ------
    - A small effect size is around 0.2, medium is 0.5, and large is 0.8, 
      though these thresholds can vary based on context.
    """
    
    import statistics
    

    diff = s2.mean()-s1.mean()
    
    n1 = len(s1)
    n2 = len(s2)
    sd1 = statistics.stdev(s1)
    sd2 = statistics.stdev(s2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1**2+(n2-1)*sd2**2)/(n1+n2-2))
    
    return diff/pooled_sd