# virtuRloops Training (Python)

This is the README file for the training (Python) part of the project "virtuRloops". This project involves the analysis of R-loop mapping data and other high-throughput sequencing features using various machine learning models and tools. The code provided here performs data preprocessing, model training, and evaluation for different feature combinations and model types. It also generates visualizations such as ROC curves, PR curves, histograms, and confusion matrices.

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

The goal of this part of the project is to train feature matrices corresponding to R-loop sites from human cell lines (HEK293, HeLa, and K562). The notebook provided here performs the following tasks:

- Data loading and preprocessing.
- Feature selection and combination.
- Model training and hyperparameter tuning for different machine learning models, including Random Forest, XGBoost, AdaBoost, Logistic Regression, Gradient Boosting, MLP, Naive Bayes, and Decision Tree.
- Evaluation of model performance using metrics such as precision, recall, F1-score, accuracy, balanced accuracy, Matthews correlation coefficient, AUC-ROC, and AUC-PR.
- Visualization of model performance with ROC curves, PR curves, histograms of predicted probabilities, and confusion matrices.
- Saving the best models and results to separate files.

## Installation

To run this notebook, you'll need Python and several libraries. You can install the required libraries using pip:

```bash
pip install pandas seaborn xgboost scikit-learn matplotlib keras keras-tuner pyreadr joblib
```

Make sure you have the necessary datasets in the specified directories as mentioned in the notebook.

## Usage

The code is structured as a Jupyter Notebook that can be run interactively. To use the notebook, follow these steps:

1. Ensure you have installed the required libraries as mentioned in the Installation section.
2. Place the required datasets in the appropriate directories (e.g., `HEK293`, `HeLa`, `K562` subdirectories under the `matrices` directory).
3. Open the notebook in Jupyter or any compatible environment.
4. Execute the cells in sequence to perform data preprocessing, model training, and evaluation.
5. The notebook will generate various visualizations and save the best models and results in the output and models directories.

## Data

The data used in this project consists of biological data for three human cell lines:

- **HEK293**
- **HeLa**
- **K562**

The datasets include positive and negative polarities and are loaded from CSV files located in subdirectories under the `matrices` directory.

## Models

The notebook includes the following machine learning models:

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

The notebook generates various result files and visualizations:

- CSV files containing detailed results for each model and dataset pair in the `output` directory.
- Best model files for each model and dataset pair in the `models` directory.
- ROC curves and AUC-ROC plots in the `output` directory.
- Precision-Recall curves and AUC-PR plots in the `output` directory.
- Histograms of predicted probabilities in the `output` directory.
- Confusion matrices in the `output` directory.

## Contributing

Contributions to this project are welcome. You can contribute by improving the code, adding new features, fixing bugs, or enhancing the documentation. Please follow the standard code of conduct and open a pull request for your contributions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
