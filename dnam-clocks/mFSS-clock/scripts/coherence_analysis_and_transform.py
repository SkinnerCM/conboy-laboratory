
"""
------------------------------------------------------------------------------
 Author: Colin M. Skinner
 Date Created: 2024-08-02
 Last Modified: 2024-11-25
 Description:   

 Dependencies:  - Python 3.8
                - scikit-learn: For machine learning model and metrics
                - SciPy: For statistical analysis (calculating correlation coefficients)
                - pandas (assumed): For handling data in DataFrame format
                - numpy : For handling data in array format

 Usage:         

 Notes:         
 ------------------------------------------------------------------------------
"""

def get_line(params):
    
    x_vals = np.linspace(0,99,100,dtype=int)
    
    line = [(x,params[0]*x+params[1]) for x in x_vals]
    
    return line    


def calculate_distance(line1, line2):
    """
    Calculate the average distance between corresponding points on two lines.
    
    Parameters:
    line1 (list of tuples): List of (x, y) coordinates representing the first line.
    line2 (list of tuples): List of (x, y) coordinates representing the second line.
    
    Returns:
    float: Average distance between corresponding points on the two lines.
    """
    # Ensure the lines have the same number of points
    assert len(line1) == len(line2), "Lines must have the same number of points"
    
    # Calculate the distances between corresponding points
    distances = [p1[1] - p2[1] for p1, p2 in zip(line1, line2)]
    
    # Calculate the average distance
    average_distance = np.mean(distances)
    
    return average_distance

def get_shifts(data, meta, cg_set, healthy_tag):    

    up_shift=[]
    down_shift=[]

    healthy = meta.disease_state==healthy_tag
    disease = meta.disease_state!=healthy_tag


    meta_h = meta[healthy]
    meta_d = meta[disease]
    data_h = data[healthy]
    data_d = data[disease]

    for i, cg in enumerate(cg_set):

        h_regression = stats.linregress(meta_h.age, data_h[cg].tolist())
        h_slope, h_intercept, h_rvalue, pvalue, h_stderr = h_regression
        
        h_line = get_line((h_slope,h_intercept))

        d_regression = stats.linregress(meta_d.age, data_d[cg].tolist())
        d_slope, d_intercept, d_rvalue, pvalue, d_stderr = d_regression
        
        d_line = get_line((d_slope,d_intercept))
        
        d = conditioned_cohens(h_line, d_line, h_stderr, d_stderr)


        if h_slope > 0:
            
            avg_dist = calculate_distance(d_line,h_line)
            

            if avg_dist>0:
                up_shift+=[(cg, avg_dist, h_rvalue, d, 1,i)]            
            else:
                down_shift+=[(cg, avg_dist, h_rvalue, d, -1,i)]
        else:
            
            avg_dist = calculate_distance(d_line,h_line)
            if avg_dist>0:
                up_shift+=[(cg, avg_dist, h_rvalue, d, 1,i)]
            else:
                down_shift+=[(cg, avg_dist,h_rvalue, d, -1,i)]
                
    up_shift = pd.DataFrame(up_shift, columns=['CpG', 'Shift', 'Correlation','Cohens d','Sign','Order'])
    up_shift.reset_index(inplace=True,drop=True)
#     up_shift = up_shift.sort_values(by='Shift', ascending=False, ignore_index=True)
    
    down_shift = pd.DataFrame(down_shift, columns=['CpG', 'Shift', 'Correlation','Cohens d','Sign', 'Order'])
    down_shift.reset_index(inplace=True,drop=True)
#     down_shift = down_shift.sort_values(by='Shift', ascending=True, ignore_index=True)
                
    return up_shift, down_shift



def get_coherence(dataset, meta, model, healthy_tag):
    
    up_shift, down_shift = get_shifts(dataset, meta, model.CpG, healthy_tag)
    
    down_shift['Weight'] = model[model['CpG'].isin(down_shift.CpG)].Weight.tolist()
    down_shift['Coherence'] = (down_shift.Shift*down_shift.Weight)>0
    
    up_shift['Weight'] = model[model['CpG'].isin(up_shift.CpG)].Weight.tolist()
    up_shift['Coherence'] = (up_shift.Shift*up_shift.Weight)>0
    
    
    total_coherence = pd.concat([up_shift, down_shift], ignore_index=True)
    
    print('Down-shifted coherence: :', down_shift.Coherence.sum()/len(down_shift))
    print('Down-shifted mean weight: ', abs(down_shift.Weight).mean())
    print('Up-shifted coherence: :', up_shift.Coherence.sum()/len(up_shift))
    print('Up-shifted mean weight: ', abs(up_shift.Weight).mean())
    print('Total coherence: ', (down_shift.Coherence.sum()+up_shift.Coherence.sum())/len(model.CpG))
    print('Mean coherence weight: ', abs(total_coherence[total_coherence.Coherence==True].Weight).mean())
    print('Max coherence weight: ', abs(total_coherence[total_coherence.Coherence==True].Weight).max())
    print('Mean incoherence weight: ', abs(total_coherence[total_coherence.Coherence!=True].Weight).mean())
    print('Max incoherence weight: ', abs(total_coherence[total_coherence.Coherence!=True].Weight).max())
    
    return total_coherence    

def coherence_transform(data, cg_set):
    
    transformed_df = data[cg_set.CpG.tolist()].copy()
    
    for i, cg in enumerate(cg_set.CpG.tolist()):
        
        if cg_set.Shift[i]<0:
            transformed_df[cg] = 1 - transformed_df[cg]
            
    return transformed_df


def conditioned_cohens(line1, line2, sd1, sd2):
    
    """This function calculates the effect size between two regression lines. Takes in ordered
       lists for the y-values of the regression lines, as well as the individual standard errors for the 
       two regressions
    """
    
    import statistics
    

    diff = calculate_distance(line1, line2)
    
    n1 = len(line1)
    n2 = len(line2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1+(n2-1)*sd2)/(n1+n2-2))
    
    return diff/pooled_sd