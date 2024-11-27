"""
--------------------------------------------------------------------------------------------------------------
 File Name:     figure_plots.py
 Author:        Colin M. Skinner
 Date Created:  2024-08-02
 Last Modified: 2024-11-26

 Description:   This script provides functions for generating and visualizing statistical plots to evaluate machine learning 
                model performance on age prediction tasks. The plots include scatter plots, residual analyses, and comparisons 
                of predictions between control and disease groups, with annotated statistical significance.

 Dependencies:  - Python 3.8
                - seaborn
                - matplotlib
                - scipy
                - pandas
                - sklearn (for metrics)

 Usage:         - The `fig3_plots` function generates plots for a given model and dataset, focusing on actual vs. predicted age 
                  comparisons, residual distributions, and boxplots by disease condition.
                - The `fig4_plots` function offers extended visualization and analysis for feature-refictified coherence models.

 Notes:         - Ensure proper preprocessing of input data before calling these functions.
                - Customize disease condition names, color palettes, and statistical methods as required.
 -------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from utils.stats_utils import cohens_d

def fig3_plots(df,df_meta, model, model_selection, d_condition, flag=False):
    
    """
    Generates a series of plots for a machine learning model's predictions 
    against actual values, including scatter plots, residual distributions, 
    and boxplots, while calculating statistical significance and effect sizes.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data containing features used by the model.
    df_meta : pandas.DataFrame
        Metadata associated with the dataset, including age and health condition.
    model : object
        A machine learning model that implements a `predict` method for inference.
    model_selection : list
        A list of feature names used for predictions by the model.
    d_condition : str
        Name of the disease condition for comparison against a control group.
    flag : bool, optional
        If `True`, uses `simple_disease_state` for coloring in the plots; 
        otherwise, uses `disease_state`. Default is `False`.

    Outputs:
    --------
    - Scatter plot of actual age vs. predicted age, with linear regression stats.
    - KDE plot of residuals, annotated with effect size and p-value.
    - Boxplot of residuals by health condition, annotated with significance stats.

    Calculations:
    -------------
    - Computes regression statistics for predicted vs. actual age.
    - Calculates residuals (predicted - actual) for each sample.
    - Performs Welch's t-test and computes Cohen's d effect size between groups.

    Notes:
    ------
    - Residuals are plotted for both control and disease conditions.
    - Visuals use customized color palettes and ordering of categories.
    """
    
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



def fig4_plots(df,df_meta, fsw_selection, reference, method, d_condition, flag=False):

    """
    Generate a series of plots for visualizing model predictions, residuals, and statistical analysis on age predictions.

    This function creates three plots:
    1. A scatter plot comparing actual vs predicted ages with regression statistics.
    2. A histogram of residuals for the healthy vs diseased cohorts, with density estimation and statistical annotations.
    3. A boxplot of residuals for healthy vs diseased cohorts with statistical significance indicated.

    Parameters:
    -----------
    df : pd.DataFrame
        The data frame containing the features used for prediction.
    df_meta : pd.DataFrame
        The metadata frame containing age, disease state, and other relevant information.
    fsw_selection : list
        The list of feature selection columns to use in the model prediction.
    reference : str
        The reference label for the healthy group (typically 'Control').
    method : str
        The method used for the model (e.g., 'fsw_model').
    d_condition : str
        The disease state to compare with the control group.
    flag : bool, optional, default=False
        Whether to use a simple disease state for the hue in the plot.

    Returns:
    --------
    None
        The function generates and displays the following plots:
        1. Scatter plot of actual vs predicted age.
        2. Histogram plot of residuals with statistical annotations.
        3. Boxplot of residuals with significance line.

    Notes:
    ------
    - The scatter plot includes a line of identity and regression statistics.
    - The histogram plot includes KDEs and calculates effect size (Cohen's d) and p-value from a Welch's t-test.
    - The boxplot compares the residuals of healthy vs diseased cohorts.
    """
    
    if flag:
        hue_val = df_meta.simple_disease_state
    else:
        hue_val = df_meta.disease_state
    
    df_transform = coherence_transform(df, d_shift)
    df_preds = fsw_model.predict(df_transform[fsw_selection])
    df_meta['fsw preds'] = df_preds

    regression = stats.linregress(df_meta[df_meta.healthy==0].age, df_preds[df_meta.healthy==0])
    slope, intercept, rvalue, pvalue, stderr = regression

    mae = metrics.mean_absolute_error(df_meta.age, df_preds)


    conditions = ['Control', d_condition]
     # Define the desired order of categories
    category_order = [conditions[0], conditions[1]]

    blue = sns.color_palette()[0]
    red=sns.color_palette()[3]

    custom_palette = {conditions[0]:blue, conditions[1]:red}


    plt.figure(figsize=(10, 8)) 
    # Create a scatter plot of the transformed data
    sns.scatterplot(x=df_meta.age, y=df_preds, hue = hue_val, palette=custom_palette, hue_order=category_order, alpha=0.75, s=150, edgecolor="k")
    vals = np.linspace(1,100,100)
    sns.lineplot(x=vals, y=vals,linewidth=3, color='k',linestyle='--')
    plt.xlabel('Actual age (yrs)',fontsize=30)
    plt.ylabel('Predicted',fontsize=30)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize='xx-large')

    plt.title('{}: Control r={:.2f}, p={:.1g}, Control MAE={:.1f} yrs'.format(df_meta.series_id[0], rvalue, pvalue, mae),fontsize=20)

    
    df_meta['Residuals'] = df_meta['fsw preds'] - df_meta.age

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
    plt.title('{}'.format(df_meta.series_id[0], rvalue, pvalue, mae),fontsize=20)
    plt.ylabel('Residuals (yrs)',fontsize=30)
