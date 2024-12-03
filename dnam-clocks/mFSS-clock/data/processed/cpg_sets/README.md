# File Descriptions:

*hannum_age_corrs.xlsx*: This spreadsheet contains the R-squared values and standard errors for the correlation between each CpG site and sample ages in the Hannum dataset (GSE40279, N=656). The beta values for each CpG were regressed on sample age using the get_age_corrs function, which is implemented in stats_utils.py.

*filtered_age_corrs.xlsx*: This spreadsheet was derived from hannum_age_corrs by first filtering out any SNP-associated CpGs (with reference to the Illumina HumanMethylation450 BeadChip manifest). Next, it was filtered on the intersection of the CpGs in the hannum32.pkl (GSE40279) and lehne32_reduced.npy (GSE55763) datasets.  