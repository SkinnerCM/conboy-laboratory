"""
---------------------------------------------------------------------------------------------
 Author: Colin M. Skinner
 Date Created: 2024-08-02
 Last Modified: 2024-11-17

 Description:   This module provides statistical analysis functions tailored for research 
                involving biological data. It includes utilities for calculating effect 
                sizes (Cohen's d) and assessing correlations between features (e.g., CpG 
                sites) and metadata such as age   
                
 Dependencies:  - Python 3.8
                - pandas (for data manipulation)
                - numpy (for numerical computations)
                - statistics (for computing standard deviations and other metrics)
                - scipy.stats (for linear regression calculations)                  

 Usage:         Import `statistics_analysis.py` into a Jupyter Notebook or Python environment.
                Use the following functions as needed:
                - `cohens_d`: Computes Cohen's d for effect size estimation between two groups.
                - `get_age_corrs`: Calculates R-squared correlations and standard errors between
                   features and a specified metadata field (e.g., age).

 Notes:         - `cohens_d` assumes input data as numeric lists, arrays, or pandas Series.
                - `get_age_corrs` requires a DataFrame of features (e.g., CpG sites) and a 
                  metadata DataFrame with an 'age' column.
                - Ensure input data is cleaned and formatted correctly to avoid runtime errors.
                - Correlation results are sorted by R-squared values in descending order for
                  easy interpretation.
 --------------------------------------------------------------------------------------------
 """


import statistics
import pandas as pd
import numpy as np

def cohens_d(s1, s2):
      

    diff = s2.mean()-s1.mean()
    
    n1 = len(s1)
    n2 = len(s2)
    sd1 = statistics.stdev(s1)
    sd2 = statistics.stdev(s2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1**2+(n2-1)*sd2**2)/(n1+n2-2))
    
    return diff/pooled_sd


def get_age_corrs(df, meta):

    corrs = []

    for cg in df.columns:

        #regress a given predictor on age
        regression = stats.linregress(meta.age.astype(float), df[cg])
        slope, intercept, rvalue, pvalue, stderr = regression

        corrs+=[(cg, rvalue**2, stderr)]

    corrs = pd.DataFrame(corrs, columns=['CpG', 'R-squared', 'Stderr'])
    corrs.sort_values('R-squared', inplace=True, ascending=False)

    return corrs