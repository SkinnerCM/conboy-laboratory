
"""
------------------------------------------------------------------------------
 Author: Colin M. Skinner
 Date Created: 2024-08-02
 Last Modified: 2024-11-24
 Description:   This script provides a set of functions to train and evaluate DNA methylation
                clocks using the mFSS (modified Forward Stepwise Selection) algorithm.
                Specifically, it includes functionality to fit a Linear Regression model to
                predict age based on CpG site methylation data, perform stepwise feature
                selection, and evaluate the model on validation and test datasets.

 Dependencies:  - Python 3.8
                - scikit-learn: For machine learning model and metrics
                - SciPy: For statistical analysis (calculating correlation coefficients)
                - pandas (assumed): For handling data in DataFrame format
                - numpy : For handling data in array format

 Usage:         Import `mfss_model_train.py` into a Jupyter Notebook or other
                Python environment and call the desired functions to train a clock with the mFSS algorithm.


 Notes:         - The `fit_and_evaluate_model` function trains a Linear Regression model, with
                  the option to impose positive coefficients if desired.
                - The `mfss_ols` function applies the modified forward stepwise selection algorithm on a list
                  of CpG sites, iteratively adding features to improve model performance on 
                  validation data.
                - The script assumes age as the target variable in the training labels.
 ------------------------------------------------------------------------------
 """


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt

from utils.stats_utils import cohens_d

def fit_and_evaluate_model(x_train, y_train, x_val, y_val, x_test, test_labels, pos_weights=False):
    """
    Fit a Linear Regression model and evaluate performance on validation and test sets.

    Parameters:
    - x_train: Training features
    - y_train: Training labels
    - x_val: Validation features
    - y_val: Validation labels
    - x_test: Test features
    - test_labels: Test labels
    - pos_weights: Whether to use positive constraints in Linear Regression

    Returns:
    - val_mse: Mean squared error on validation set
    - test_mse: Mean squared error on test set
    - val_r_val: Correlation coefficient on validation set
    - test_r_val: Correlation coefficient on test set
    """
    regr = LinearRegression(positive=pos_weights, n_jobs=-1).fit(x_train, y_train)
    val_preds = regr.predict(x_val)
    test_preds = regr.predict(x_test)

    # Calculate performance metrics
    val_mse = mean_squared_error(y_val, val_preds)
    test_mse = mean_squared_error(test_labels, test_preds)

    # Calculate r-values
    val_r_val = compute_r_value(val_preds, y_val)
    test_r_val = compute_r_value(test_preds, test_labels)

    return val_mse, test_mse, val_r_val, test_r_val

def compute_r_value(preds, actuals):
    """
    Compute the correlation coefficient between predictions and actual values.
    
    Parameters:
    - preds: Predicted values
    - actuals: Actual values

    Returns:
    - r_value: Correlation coefficient
    """
    if len(set(preds)) < 2 or len(set(actuals)) < 2:
        return 0
    return stats.linregress(preds, actuals).rvalue


def get_age_corrs(df, meta):
    
    """
    Calculate the R-squared values and standard errors of CpG methylation levels
    with respect to age using linear regression.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing methylation levels where rows are samples 
        and columns are CpG sites.
    meta : pandas.DataFrame
        A metadata DataFrame that includes an 'age' column corresponding 
        to the age of each sample in `df`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the CpG site names, R-squared values, and 
        standard errors, sorted in descending order of R-squared values.

    Notes
    -----
    - Each CpG site's methylation values are regressed on age using simple 
      linear regression.
    - The R-squared value indicates the proportion of variance in methylation 
      explained by age.
    - The standard error reflects the precision of the slope estimate for each 
      CpG site.
    - Sorting by R-squared highlights CpG sites most strongly correlated 
      with age.

    Example
    -------
    >>> import pandas as pd
    >>> from scipy import stats
    >>> methylation_data = pd.DataFrame({
    ...     'cg1': [0.5, 0.6, 0.55],
    ...     'cg2': [0.7, 0.8, 0.75]
    ... })
    >>> metadata = pd.DataFrame({'age': [25, 30, 35]})
    >>> get_age_corrs(methylation_data, metadata)
            CpG  R-squared    Stderr
    0      cg2   0.900000  0.028868
    1      cg1   0.800000  0.028868
    """
    
    corrs = []
    
    for cg in df.columns:
        
        #regress a given predictor on age
        regression = stats.linregress(meta.age.astype(float), df[cg])
        slope, intercept, rvalue, pvalue, stderr = regression
        
        corrs+=[(cg, rvalue**2, stderr)]
        
    corrs = pd.DataFrame(corrs, columns=['CpG', 'R-squared', 'Stderr'])
    corrs.sort_values('R-squared', inplace=True, ascending=False)
    
    return corrs



def mfss_ols(cg_list, train, train_labels, test, test_labels, threshold, pos_weights=False):
    """
    Perform stepwise model selection.

    Parameters:
    - cg_list: List of CpG sites
    - train: Training data
    - train_labels: Labels for training data
    - test: Test data
    - test_labels: Labels for test data
    - threshold: Number of unsuccessful iterations before stopping
    - pos_weights: Whether to use positive constraints in Linear Regression

    Returns:
    - model_cgs: List of selected CpG sites
    - best_iter: Iteration number of the best model
    - val_mse: List of validation mean squared errors
    - val_r_val: List of validation r-values
    - test_mse: List of test mean squared errors
    - test_r_val: List of test r-values
    """
    countdown = threshold
    iteration = 0
    best_mse = float('inf')
    model_cgs = []
    val_mse = []
    val_r_val = []
    best_iter = 0
    test_mse = []
    test_r_val = []

    while countdown > 0 and iteration < len(cg_list):
        print(f'Iteration: {iteration}')
        
        # Update model with the next CpG site
        model_cgs.append(cg_list[iteration])
        temp_data = train[model_cgs]

        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(temp_data, train_labels.age,
                                                          test_size=0.15, random_state=42)
        
        # Fit and evaluate model
        curr_val_mse, curr_test_mse, curr_val_r_val, curr_test_r_val = fit_and_evaluate_model(
            x_train, y_train, x_val, y_val, test[model_cgs], test_labels.age, pos_weights
        )
        
        # Store performance metrics
        val_mse.append(curr_val_mse)
        test_mse.append(curr_test_mse)
        val_r_val.append(curr_val_r_val)
        test_r_val.append(curr_test_r_val)

        # Update best model if current model is better
        if curr_test_mse < best_mse:
            best_mse = curr_test_mse
            best_iter = iteration
            countdown = threshold
        else:
            countdown -= 1
        
        iteration += 1
        
    return model_cgs, best_iter, val_mse, val_r_val, test_mse, test_r_val



def fig3_plots(df,df_meta, model, model_selection, d_condition, flag=False):
    
    if flag:
        hue_val = df_meta.simple_disease_state
    else:
        hue_val = df_meta.disease_state

    preds = model.predict(df[model_selection])
    df_meta['model preds'] = preds

    regression = stats.linregress(df_meta.age, preds)
    slope, intercept, rvalue, pvalue, stderr = regression

    mae = metrics.mean_absolute_error(df_meta.age, preds)


    conditions = ['Control', d_condition]
    # Define the desired order of categories
    category_order = [conditions[0], conditions[1]]

    blue = sns.color_palette()[0]
    red=sns.color_palette()[3]

    custom_palette = {conditions[0]:blue, conditions[1]:red}

    # Create a scatter plot of the transformed data
    plt.figure(figsize=(10, 8)) 
    
    sns.scatterplot(x=df_meta.age, y=preds, hue = hue_val, 
                    palette=custom_palette, hue_order=category_order,
                    alpha=0.75, s=150, edgecolor="k")
    
    vals = np.linspace(1,100,100)
    sns.lineplot(x=vals, y=vals,linewidth=3, color='k',linestyle='--')
    plt.xlabel('Actual age (yrs)',fontsize=30)
    plt.ylabel('Predicted',fontsize=30)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize='xx-large')

    plt.title('{}: r={:.2f}, p={:.1g}, MAE={:.1f} yrs'.format(df_meta.series_id[0], rvalue, pvalue, mae),fontsize=20)
        
    

    
    #create kdeplot
    df_meta['Residuals'] = df_meta['model preds'] - df_meta.age
    
    from scipy.stats import ttest_ind
    h_resids = df_meta[hue_val=='Control'].Residuals
    d_resids = df_meta[hue_val!='Control'].Residuals
    # Perform the Welch's t-test (indfendent t-test with unequal variances)
    statistic, p_value = ttest_ind(h_resids, d_resids, equal_var=False)
    effect_size =  cohens_d(h_resids, d_resids)

    plt.figure(figsize=(10, 8)) 


    sns.histplot(
            data=df_meta,
            x='Residuals',
            hue=hue_val,
            palette=custom_palette,
            bins=20,
            kde=True,
            alpha=0.4,
            element="step",
            legend=False,
            stat = 'density',
            common_norm=False,
            line_kws={'linewidth': 4}
            )
    plt.xlabel('Residual (yrs)', fontsize=28)
    plt.xticks(fontsize=24)
    plt.ylabel('Density', fontsize=28)
    plt.yticks(fontsize=24)

    # Plot means for each cohort (healthy status)
    mean_healthy = h_resids.mean()
    mean_unhealthy = d_resids.mean()

    plt.axvline(x=mean_healthy, color=blue, linestyle='--', linewidth=3, 
               ymin=0.01, ymax=0.95, dashes=(4.5, 4.28), zorder=6)
    plt.axvline(x=mean_healthy, color='k', linestyle='--', linewidth=5, ymax=0.95)

    plt.axvline(x=mean_unhealthy, color=red, linestyle='--', linewidth=3, 
                 ymin=0.01, ymax=0.95, dashes=(4.5, 4.28), zorder=6)
    plt.axvline(x=mean_unhealthy, color='k', linestyle='--', linewidth=5, ymax=0.95)

    plt.text(0.65, 0.95, f"d={effect_size:.2f}", fontsize=26, verticalalignment='top', 
         horizontalalignment='left', transform=plt.gca().transAxes);
    plt.text(0.65, 0.85, f"p={p_value:.1e}", fontsize=26,  verticalalignment='top', 
         horizontalalignment='left', transform=plt.gca().transAxes);
    


    #Create boxplot
    plt.figure(figsize=(10, 8)) 
    
    sns.boxplot(data=df_meta, x=hue_val, y='Residuals', palette=custom_palette, order=category_order,linewidth=3)
    # Add significance bar and p-value

    max_val = max(max(h_resids),max(d_resids))
    min_val = min(min(h_resids),min(d_resids))

    plt.plot([0, 1], [max_val+2, max_val+2], 'k-', lw=2)
    plt.text(0.5, max_val+3, f'd = {effect_size:.2g} \n p = {p_value:.2g}', ha='center', fontsize=20)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim([min_val-5, max_val+15])

    plt.xlabel(None)
    plt.ylabel('Residuals (yrs)',fontsize=30)
