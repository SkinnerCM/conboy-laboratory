# Dissertation Repository: Hypothesis-driven training for DNA Methylation Clocks and Feature Rectification for Linear Model Coherence to Detect Inflammaging

Welcome to the repository for my dissertation, **"Hypothesis-driven training for DNA Methylation Clocks and Feature Rectification for Linear Model Coherence to Detect Inflammaging"**, submitted in partial fulfillment of the requirements for the Joint Doctor of Philosophy with University of California, San Francisco in Bioengineering and the Designated Emphasis in Computational and Genomic Biology in the Graduate Division of the University of California, Berkeley. This repository contains all the code, figures, and Jupyter notebooks used to generate the results and figures presented in the dissertation.

## Repository Contents

### 1. **Code**
   - **`src/`**: Python scripts and utility functions used for data processing, analysis, and model training.
     - Example: `feature_selection.py` - functions for analyzing CpG feature importance in DNA methylation clocks.
   - **Dependencies**:
       - python=3.8.16
       - pip:
    	- numpy==1.24.4
    	- pandas==1.5.3
    	- scipy==1.10.0
    	- scikit-learn==1.2.0
    	- matplotlib==3.6.3
    	- seaborn==0.12.2
    	- jupyterlab==3.6.1
    	- ipykernel==6.20.2
        - notebook==6.5.2

### 2. **Figures**
   - **`figures/`**: Contains all the figures used in the dissertation, including:
     - High-resolution images for publication.
     - Supplementary and exploratory visualizations.
     - Subdirectories grouped by chapter, e.g., `chapter1/`, `chapter2/`.

### 3. **Notebooks**
   - **`notebooks/`**: Jupyter notebooks for data preprocessing, analysis, and figure generation.
     - Example notebooks:
       - `Chapter1.ipynb`: Analysis of the biological relevance of CpG sites across DNA methylation clocks and exploration of prediction variances in biological age estimation.

### 4. **Data**
   - **`data/`**: Processed datasets used for analysis. 
   - Instructions for accessing raw data can be found in `data/README.md`.

## How to Use This Repository

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SkinnerCM/conboy-laboratory.git
   cd dissertation
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Run analyses**:
   - Navigate to the relevant notebook in the notebooks/ directory or use the command line to execute scripts from the src/ directory.
4. **Reproduce figures**:
   Open the corresponding notebook in the notebooks/ directory or run the scripts listed in the src/ directory.

## License

© Colin M. Skinner, 2024. All rights reserved.

This work is the intellectual property of Colin M. Skinner. Redistribution, modification, or commercial use is prohibited without explicit permission.

For inquiries about permissions or collaboration, please contact skinner.colinm@gmail.com.

## Citation
If you use this repository in your work, please cite:

Colin M. Skinner. "Hypothesis-driven training for DNA Methylation Clocks and Feature Rectification for Linear Model Coherence to Detect Inflammaging." Ph.D. Dissertation, University of California, Berkeley, 2025. (\<DOI or link\>)
