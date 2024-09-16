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