# virtuRloops Training (Python)

This is the README file for the training (Python) part of the project "virtuRloops". This project involves the analysis of biological data using various machine learning models and tools. The code provided here performs data preprocessing, model training, and evaluation for different feature combinations and model types. It also generates visualizations such as ROC curves, PR curves, histograms, and confusion matrices.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The goal of this part of the project is to analyze biological data related to two cell types, 3T3 and E14, with both positive and negative polarities. The code provided here performs the following tasks:

- Data loading and preprocessing.
- Feature selection and combination.
- Model training and hyperparameter tuning for different machine learning models, including Random Forest, XGBoost, AdaBoost, Logistic Regression, Gradient Boosting, MLP, Naive Bayes, and Decision Tree.
- Evaluation of model performance using metrics such as precision, recall, F1-score, accuracy, balanced accuracy, Matthews correlation coefficient, AUC-ROC, and AUC-PR.
- Visualization of model performance with ROC curves, PR curves, histograms of predicted probabilities, and confusion matrices.
- Saving the best models and results to separate files.

## Installation

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

## Contributing
Contributions to this project are welcome. You can contribute by improving the code, adding new features, fixing bugs, or enhancing the documentation. Please follow the standard code of conduct and open a pull request for your contributions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.