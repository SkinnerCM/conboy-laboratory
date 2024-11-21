Series GSE55763		Query DataSets for GSE55763
Status	Public on Jan 22, 2015
Title	A coherent approach for analysis of the Illumina HumanMethylation450 BeadChip improves data quality and performance in epigenome-wide association studies
Organism	Homo sapiens
Experiment type	Methylation profiling by array
Summary	We developed a comprehensive analysis pipeline to conduct Epigenome-wide Association Studies (EWAS) using the Illumina Infinium HumanMethylation450 BeadChip, based on data from 2,664 individuals, and 36 samples measured in duplicate. We propose new approaches to quality control, data normalisation and batch correction through control-probe adjustment, and demonstrate that these improve data-quality. Using permutation testing we establish a null hypothesis for EWAS, show how it can be affected by correlation between individual methylation markers and present methods to restore statistical independence.
 	
Overall design	Bisulfite-converted DNA for 2,664 human samples and 36 technical replicates were hybridised to the Illumina Infinium 450k Human Methylation Beadchip v1.2.
 	
Contributor(s)	Lehne B, Drong A, Loh M, Zhang W, Scott WR, Tan ST, Afzal U, Scott J, Jarvelin MR, Elliott P, McCarthy MI, Kooner JS, Chambers JC
Citation(s)	
Lehne B, Drong AW, Loh M, Zhang W et al. A coherent approach for analysis of the Illumina HumanMethylation450 BeadChip improves data quality and performance in epigenome-wide association studies. Genome Biol 2015 Feb 15;16(1):37. PMID: 25853392
Wahl S, Drong A, Lehne B, Loh M et al. Epigenome-wide association study of body mass index, and the adverse outcomes of adiposity. Nature 2017 Jan 5;541(7635):81-86. PMID: 28002404


***Notes***

The dataset contained technical replicates which were removed from the "Lehne_reduced.pkl" file. To create this, I loaded the "Lehne.pkl" and "Lehne_pmeta.xlsx" files and used the keys from the pmeta file to extract the population study samples from "Lehne.pkl". Next, the columns were sorted alpha-numerically in ascending order and the resulting dataframe was saved as "Lehne_reduced.pkl" 