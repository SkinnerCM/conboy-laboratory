import pandas as pd

def get_gene_list(manifest, cg_set):
	"""Get a list of genes annotated to the provided CpGs."""
	#get the list of genes annotated to the basic mfss cgs
	return manifest[manifest['Name'].isin(cg_set)]['UCSC_RefGene_Name'].tolist()


def make_gene_df(gene_list, cg_set):
	"""Create a DataFrame linking probes to their annotated genes."""

	gene_df = pd.DataFrame({'Probe': cg_set, 'Gene': gene_list})

	# Iterate over each cell in the column
	for index, row in gene_df.iterrows():
	    gene_list_str = row['Gene']
	    
	    # Skip if the value is NaN
	    if pd.isnull(gene_list_str):
	        continue
	    
	    # Split the string by ';', remove duplicates, and join into a single string
	    unique_genes = ';'.join(set(gene_list_str.split(';')))
	    
	    # Replace the value in the dataframe with the unique genes
	    gene_df.at[index, 'Gene'] = unique_genes
	    #drop the probes with nans
	gene_df = gene_df[~gene_df.Gene.isna()]
	gene_df = gene_df.set_index('Probe')	

	return gene_df

def gene_sep(gene_df):
	"""Separate genes into a final DataFrame with associated probes."""
	df = pd.DataFrame({'gene_list': gene_df.Gene})

	# Initialize an empty dictionary to store gene-probe associations
	gene_probe_dict = {}

	# Iterate over each row in the dataframe
	for index, row in df.iterrows():
	    gene_list_str = row['gene_list']
	    
	    # Skip if the value is NaN
	    if pd.isnull(gene_list_str):
	        continue
	    
	    # Split the string by ';' to get a list of genes
	    genes = gene_list_str.split(';')
	    
	    # Iterate over each gene
	    for gene in genes:
	        # Add the gene to the dictionary if not already present
	        if gene not in gene_probe_dict:
	            gene_probe_dict[gene] = set()
	        # Add the probe to the set of probes associated with the gene
	        gene_probe_dict[gene].add(index)

	final_df = pd.DataFrame([(', '.join(probes), gene) for gene, probes in gene_probe_dict.items()],columns=['Probes', 'Gene'])

	return final_df

def get_gene_annotations(manifest, cg_set):
    """
    This function orchestrates all steps to process CpGs, retrieve gene annotations,
    and return a DataFrame with unique genes and associated probes.

    Parameters:
        manifest (pd.DataFrame): The manifest containing CpG information and gene annotations.
        cg_set (list): A list of CpGs for the DNA clock model.

    Returns:
        pd.DataFrame: A DataFrame with 'Probes' and 'Gene' columns.
    """
    
    # Step 1: Get the list of genes annotated to the provided CpGs
    gene_list = get_gene_list(manifest, cg_set)
        
    # Step 2: Create a DataFrame linking probes to their annotated genes
    gene_df = make_gene_df(gene_list, cg_set)
    
    # Step 3: Separate genes into a final DataFrame with associated probes
    final_df = gene_sep(gene_df)
    
    return final_df


