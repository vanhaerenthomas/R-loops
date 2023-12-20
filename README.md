
## Introduction

R-loops are three-stranded nucleic acid structures consisting of an RNA–DNA hybrid and a displaced single-stranded DNA that frequently occur during transcription from bacteria to mammals. Current methods for R-loop site prediction are based on DNA sequence, which precludes detection across cell types, tissues or developmental stages. Here we provide virtuRloops, a computational tool that allows the prediction of cell type specific R-loops using genomic features. This method is described in our study entitled [*A computational framework for the genomic prediction of cell type specific R-loops*](https://www.biorxiv.org/XXX) (currently in bioRxiv). Human and mouse predictions can be downloaded and visualized in the [*following link*](http://193.147.188.155/pmargar/drip_pred/). 

## Usage

virtuRloops provides pre-computed learning models that were trained using multiple combinations of sequencing features (see models directory). One of these model should be passed to predict R-loop sites, together with a tabulated file containing the paths of the corresponding feature files (in 4 columns bedgraph format; see features directory). Note that the program assumes that feature files are RPM normalized. If a target peak file (in 3 columns bed format) is provided, predictions will be performed on such coordinates. Peaks will be resized to 500 bp if they don't have such size. If no peak file is provided, either the mouse (mm9) or human (hg38) genomes will be used as target. Note that, since GitHub limits the size of files, we do not directly provide the corresponding whole genome bed files for these species. To compute whole genome predictions, download the file you are interested in from the following links and put it on the 'genomes' folder: [*hg38*](http://193.147.188.155/pmargar/drip_pred/hg38_1_to_22_XYM_sliding_500bp_MAPPABLE.bed), [*mm9*](http://193.147.188.155/pmargar/drip_pred/mm9_20_longest_sliding_500bp_MAPPABLE.bed).

```
Usage: ./predict [options]

Options:
	-n CHARACTER, --name=CHARACTER
		Execution name (predictions will be reported in a file with this name)

	-d CHARACTER, --datasets=CHARACTER
		Tabulated file (with no header line) containing paths of sequencing datasets (RPM normalized and 4-columns bedgraph format; check 'features' directory).
		Note that model features should be provided and others will be ignored.
		Possible feature names are: H3K9ac,H3K9me3,H3K27me3,H3K4me3,H3K36me3,H3K27ac,H3K4me1,RNA_seq,DNase,GRO_seq

	-t CHARACTER, --target=CHARACTER
		Genome assembly name or input bed file with no header. If bed file is supplied, R-loop model will be applied to its corresponding loci.
		Otherwise, genome wide predictions will be generated (only 'mm9' and 'hg38' are supported)

	-m CHARACTER, --model=CHARACTER
		R-loop model file in RData format (check 'models' directory)

	-h, --help
		Show this help message and exit
```
Calling example for given peak file:
```
./predict -n example1 -d features/features_adrenal_gland_chr22.csv -t peaks/peaks_example_chr22.bed -m models/DRIP_E14_3T3_comb/RNA_DNase_K4me3_K9me3_K36_K4me1/model.RData
```
Calling example for whole genome prediction (needs a user provided features table):
```
./predict -n example2 -d myFeatures.csv -t hg38 -m models/DRIP_E14_3T3_comb/RNA_DNase_K4me3_K9me3_K36_K4me1/model.RData
```

## Requirements

- [bedtools v2.26.0](https://bedtools.readthedocs.io/en/latest/)
- [R >= 3.6](https://cran.r-project.org/)

### R libraries:

- [optparse](https://cran.r-project.org/web/packages/optparse/index.html)
- [stringi](https://cran.r-project.org/web/packages/stringi/index.html)
- [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html)

## Python Description

The goal of the Python code is to analyze biological data related to two cell types, 3T3 and E14, with both positive and negative polarities. The code provided here performs the following tasks:

- Data loading and preprocessing.
- Feature selection and combination.
- Model training and hyperparameter tuning for different machine learning models, including Random Forest, XGBoost, AdaBoost, Logistic Regression, Gradient Boosting, MLP, Naive Bayes, and Decision Tree.
- Evaluation of model performance using metrics such as precision, recall, F1-score, accuracy, balanced accuracy, Matthews correlation coefficient, AUC-ROC, and AUC-PR.
- Visualization of model performance with ROC curves, PR curves, histograms of predicted probabilities, and confusion matrices.
- Saving the best models and results to separate files.

## Python Installation

To run this code, you'll need Python and several libraries. You can install the required libraries using pip:

```bash
pip install pandas seaborn xgboost scikit-learn matplotlib keras pyreadr joblib
```

Make sure you have the necessary datasets in the specified directories as mentioned in the code.

## Usage
The code is structured as a Python script that can be run from a Jupyter Notebook or any Python IDE. To use the code, follow these steps:

1. Ensure you have installed the required libraries as mentioned in the Installation section.

2. Make sure you have the required datasets in the specified directories (matrices).

3. Run the script, and it will perform data preprocessing, model training, and evaluation.

4. The code will generate various visualizations and save the best models and results in the output and models directories.

## Data
The data used in this project consists of biological data for two cell types, 3T3 and E14, with both positive and negative polarities. The data is loaded from CSV files located in the matrices directory.

## Models
The code includes the following machine learning models:

- Random Forest
- XGBoost
- AdaBoost
- Logistic Regression
- Gradient Boosting
- MLP (Multi-layer Perceptron)
- Naive Bayes
- Decision Tree

Each model is trained and evaluated for various hyperparameter combinations, and the best-performing model is saved.

## Results
The code generates various result files and visualizations:

- CSV files containing detailed results for each model and dataset pair in the output directory.
- Best model files for each model and dataset pair in the models directory.
- ROC curves and AUC-ROC plots in the output directory.
- Precision-Recall curves and AUC-PR plots in the output directory.
- Histograms of predicted probabilities in the output directory.
- Confusion matrices in the output directory.

