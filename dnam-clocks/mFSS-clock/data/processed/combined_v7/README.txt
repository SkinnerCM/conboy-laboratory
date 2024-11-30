Combined datasets (ordered): GSE42861 (arthritis), GSE106648 (MS), GSE52588 (Down syndrome),  
GSE100825 (Werner), GSE72774 (Parkinsons), GSE87648 (IBS)

Datasets pickled with Python version 3.8

Healthy combined dataset: All missing data were first imputed using the KNN algorithm (via sklearn's library). No normalization was 
done. The choice of K was chosen somewhat arbitrarily: K=3 for larger datasets (n>300), and K=5 for smaller (n<300). Datasets were then divided independently into healthy and disease sets. The combined healthy dataset was made two-at-a-time by taking the intersection of the CpGs of the growing "combined" datasetand the newly added dataset. The final dataset contains n=930 healthy individuals and p=446,005 CpGs. The accompanying combined metadata file has the same ordering of samples, contains the GSMs, ages, sexes, and the dataset study and GSE

Probes known to be X-reactive (Chen et al."Discovery of cross-reactive probes and polymorphic CpGs in the Illumina Infinium HumanMethylation450 microarray (2013)") were filtered from the combined datasets
	

*** Combined V3: This is a dataset of only the healthy cohorts from (ordered)

GSE42861 (arthritis), GSE73103 (obesity2), GSE52588 (Down syndrome), GSE67705 (HIV)
 
*** Combined V7: This is a dataset of both disease and healthy cohort, KNN (k=2) used to impute missing data from (ordered)

GSE42861 (arthritis_full), GSE125105 (depression_full), GSE72774 (Parkinsons Horvath), GSE106648 (MS)