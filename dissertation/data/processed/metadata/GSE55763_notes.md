# Data Processing Notes

## Dataset Overview

The dataset originally contained technical replicates which were removed in the creation of the `GSE55763_reduced.pkl` file.

## Processing Steps

1. **Load Files:**
   - Loaded the dataset from `GSE55763.pkl`.
   - Loaded metadata from `GSE55763_pmeta.xlsx`.

2. **Extract Population Study Samples:**
   - Used the keys from `GSE55763_pmeta.xlsx` to extract the relevant population study samples from `GSE55763.pkl`.

3. **Sort Data:**
   - Sorted the columns of the resulting dataframe alpha-numerically in ascending order.

4. **Save Processed Data:**
   - Saved the processed dataframe as `GSE55763_reduced.pkl`.

## Resulting File

- **`GSE55763_reduced.pkl`:** Contains the population study samples with technical replicates removed and columns sorted.