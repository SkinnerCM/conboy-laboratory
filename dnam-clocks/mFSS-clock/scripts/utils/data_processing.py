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