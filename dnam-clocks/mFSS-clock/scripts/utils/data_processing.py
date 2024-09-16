import pandas
import csv

def prep_model(model):
    """
    Prepare a model for making predictions.

    This function expects a DataFrame with two columns:
    - The first column should contain the CpG sites for the model.
    - The second column should contain the corresponding weights or importance scores.

    The function renames the columns to 'CpG' and 'Weight' and sorts the DataFrame
    by the 'CpG' column.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the model features and their weights.

    Returns:
    pd.DataFrame: The processed DataFrame with columns renamed and sorted by 'CpG'.
    """
    # Ensure the data has the correct column names
    model.columns = ['CpG', 'Weight']
    
    # Sort the data by CpG sites
    model = model.sort_values('CpG', ignore_index=True)
    
    return model


def list_writer(ur_list, file_name):

    with open(file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ur_list)
        
def list_reader(file,  encoding="utf8"):

    with open(file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[0]

    return data

def cohens_d(s1, s2):
    
    import statistics
    

    diff = s2.mean()-s1.mean()
    
    n1 = len(s1)
    n2 = len(s2)
    sd1 = statistics.stdev(s1)
    sd2 = statistics.stdev(s2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1**2+(n2-1)*sd2**2)/(n1+n2-2))
    
    return diff/pooled_sd