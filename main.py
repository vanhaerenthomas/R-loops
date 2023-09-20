import itertools, scipy, numpy as np, pandas as pd, seaborn as sns, xgboost as xgb, pyreadr, random, time, datetime as dt, matplotlib.pyplot as plt, os, json, joblib, warnings, keras, keras_tuner
from operator import itemgetter
from sklearn import metrics, preprocessing
from sklearn.metrics import (average_precision_score, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, roc_curve, auc, recall_score, precision_score, f1_score, precision_recall_curve, confusion_matrix, accuracy_score, balanced_accuracy_score, matthews_corrcoef)
from sklearn.model_selection import cross_val_score, train_test_split, ParameterGrid
from xgboost import XGBClassifier, XGBRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding
from numpy import genfromtxt
from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None 

arialfont = {'fontname':'Arial'}

import warnings

# Ignore Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

base_location = "E:\\Work\\UPO Sevilla\\Doctorate_3_DROP_3T3_E14"
matrices_path = os.path.join(base_location, "matrices")

types = ['3T3', 'E14']
dfs = {}

for t in types:
    for polarity in ['neg', 'pos']:
        file_path = os.path.join(matrices_path, f"DRIP_{t}_{polarity}.csv")
        df = pd.read_csv(file_path, sep='\t').head(10)
        df["CLASS"] = 0 if polarity == 'neg' else 1
        dfs[f"{t}_{polarity}"] = df

dfs['3T3'] = pd.concat([dfs['3T3_neg'], dfs['3T3_pos']])
dfs['E14'] = pd.concat([dfs['E14_neg'], dfs['E14_pos']])

feature_combinations = [
    ['GRO_seq'],
    ['GRO_seq', 'RNA_seq'],
    ['GRO_seq', 'RNA_seq', 'DNase'],
    ['GRO_seq', 'RNA_seq', 'DNase', 'H3K4me3'],
    ['GRO_seq', 'RNA_seq', 'DNase', 'H3K4me3', 'H3K36me3'], 
    ['GRO_seq', 'RNA_seq', 'DNase', 'H3K4me3', 'H3K36me3', 'H3K9ac', 'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K4me1']
]

for features in feature_combinations:
    features_with_class = features + ['CLASS']
    for t in types:
        df_subset = dfs[t][features_with_class]
        dfs[f"{t}_{'_'.join(features)}"] = df_subset
        dfs[f"X_train_{t}_{'_'.join(features)}"], dfs[f"X_test_{t}_{'_'.join(features)}"] = train_test_split(df_subset, test_size=0.3, shuffle=True)

# Define models
models = {
    'RF': RandomForestClassifier(random_state=42),
    'XGB': XGBClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'LogReg': LogisticRegression(random_state=42),
    'GBM': GradientBoostingClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
}

# Define parameter grids
param_grids = {
    'RF': ParameterGrid({
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }),
    'XGB': ParameterGrid({
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [6, 10, 15, 20],
        # 0.1 is optimal, skipping the rest for performance
        'learning_rate': [0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'gamma': [0, 0.5, 1.0],
        'reg_lambda': [1.0, 10.0, 50.0],
    }),
    'AdaBoost': ParameterGrid({
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.1, 0.5, 1.0]
    }),
    'LogReg': ParameterGrid({
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear'],
        'max_iter': [100, 200]
    }),
    'GBM': ParameterGrid({
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.1, 0.2, 0.3],
        'max_depth': [3, 6, 9]
    }),
    'MLP': ParameterGrid({
        'hidden_layer_sizes': [(100,), (100, 100), (200,), (200, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }),
    'NaiveBayes': ParameterGrid({}),
    'DecisionTree': ParameterGrid({
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }),
}

# Create directories for outputs if they don't exist
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

dataset_pairs = []

for features in feature_combinations:
    for t1 in types:
        for t2 in types:
            dataset_pairs.append(
                (f"{t1}_{'_'.join(features)}_to_{t2}_{'_'.join(features)}", 
                 dfs[f"X_train_{t1}_{'_'.join(features)}"], 
                 dfs[f"X_test_{t2}_{'_'.join(features)}"]
                )
            )

results = []

# Define a new results dictionary for each model
results = {model: [] for model in models.keys()}

# For storing the best model of each type for each dataset pair
pair_best_model = {pair[0]: {} for pair in dataset_pairs}

# Loop through dataset pairs
for pair_name, X_train_full, X_val_full in dataset_pairs:
    X_train, y_train = X_train_full.drop("CLASS", axis=1), X_train_full["CLASS"]
    X_val, y_val = X_val_full.drop("CLASS", axis=1), X_val_full["CLASS"]

    # Get feature names from the dataframe
    feature_names = X_train.columns.tolist()

    # Loop through models
    for model in models.keys():
        print(f"\nStarting grid search for {model} on {pair_name}")

        param_grid_size = len(param_grids[model])
        next_print_percentage = 2.5
        start_time = time.time()

        best_model = None
        best_f1 = -np.inf

        # Loop through parameter grid
        for j, params in enumerate(param_grids[model]):
            param_start_time = time.time()
            clf = models[model].__class__()
            clf.set_params(**params)
            clf.fit(X_train, y_train)
            
            X_val = np.array(X_val)

            preds = clf.predict(X_val)
            f1 = round(f1_score(y_val, preds, average='macro')*100, 2)
            precision = round(precision_score(y_val, preds, average='macro')*100, 2)
            recall = round(recall_score(y_val, preds, average='macro')*100, 2)
            accuracy = round(accuracy_score(y_val, preds)*100, 2)
            balanced_accuracy = round(balanced_accuracy_score(y_val, preds)*100, 2)
            matthews_corrcoef_score = round(matthews_corrcoef(y_val, preds)*100, 2)


            # Update best model if current model is better
            if f1 > best_f1:
                best_model = clf
                best_f1 = f1

            auc_roc = round(auc(*roc_curve(y_val, clf.predict_proba(X_val)[:, 1])[:2])*100, 2)
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, clf.predict_proba(X_val)[:, 1])
            sort_indices = np.argsort(recall_curve)
            precision_sorted = precision_curve[sort_indices]
            recall_sorted = recall_curve[sort_indices]
            auc_pr = round(auc(recall_sorted, precision_sorted)*100, 2)

            # Get importances if the model has feature_importances_ attribute
            importances = (clf.feature_importances_*100 if hasattr(clf, "feature_importances_") else None)
           
            # Create a dictionary for current results
            current_result = {
                "model": model,
                "dataset_pair": pair_name,
                "parameters": json.dumps(params),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "matthews_corrcoef": matthews_corrcoef_score,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "feature_importances": json.dumps({name: importance for name, importance in zip(feature_names, importances.tolist())}) if importances is not None else None
            }
            
            # Save results
            results[model].append(current_result)
            
            # Print status update every 2.5%
            completion_percentage = ((j + 1) / param_grid_size) * 100
            if completion_percentage >= next_print_percentage:
                elapsed_time = time.time() - param_start_time
                estimated_time_remaining = timedelta(seconds=(elapsed_time * (param_grid_size - (j + 1))))
                estimated_time_remaining_str = (datetime(1,1,1) + estimated_time_remaining).strftime('%H:%M:%S')
    
                # Split the original string into parts
                parts = pair_name.split("_to_")
    
                # Split each part further based on underscores and space out the words
                train_part = parts[0]
                test_part = parts[1]
    
                # Construct the final name
                transformed_name = f"trained on {train_part} and tested on {test_part}"
    
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model} {transformed_name}: {completion_percentage:.2f}% complete, estimated time remaining: {estimated_time_remaining_str}")
                next_print_percentage += 2.5
            
        pair_best_model[pair_name][model] = best_model
        

# Save intermediate results to CSV files per model and dataset pair
for model, model_results in results.items():
    # Before converting results to DataFrame, sort feature_importance values
    for result in model_results:
        feature_importances = result['feature_importances']
        if feature_importances is not None:
            # Parse JSON string into dict
            feature_importance_dict = json.loads(feature_importances)
            # Convert dict to list of tuples, sort by feature importance value, and convert back to dict
            feature_importance_list = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
            result['feature_importances'] = json.dumps(dict(feature_importance_list))

    # Convert results to DataFrame and sort by F1 score
    results_df = pd.DataFrame(model_results)
    results_df = results_df.sort_values(by='f1_score', ascending=False)
    csv_file_path = os.path.join("output", f"results_{model}.csv")
    results_df.to_csv(csv_file_path, index=False)
    # print(f"The CSV file for {model} was created successfully here: {os.path.abspath(csv_file_path)}")
    
# Initialize dictionary to store best model results for each pair
best_results = {}

# Loop through each model and their results
for model, model_results in results.items():
    # Convert model results to DataFrame for easier manipulation
    model_results_df = pd.DataFrame(model_results)
    
    # Loop through each dataset pair
    for pair in model_results_df['dataset_pair'].unique():
        # Filter the results for the current pair
        pair_results_df = model_results_df[model_results_df['dataset_pair'] == pair]
        
        # If there are results for this pair
        if not pair_results_df.empty:
            # Get the row with the highest F1 score
            best_result_row = pair_results_df.loc[pair_results_df['f1_score'].idxmax()]
            
            # Add the best result row to the pair's best models
            if pair not in best_results:
                best_results[pair] = []
            best_results[pair].append(best_result_row)

# Save the best models for each pair to a separate CSV file
for pair, best_models in best_results.items():
    best_models_df = pd.DataFrame(best_models)
    csv_file_path = os.path.join("output", f"best_{pair}.csv")
    best_models_df.to_csv(csv_file_path, index=False)
    # print(f"The CSV file for {pair} was created successfully here: {os.path.abspath(csv_file_path)}")

# Combine results and save to a single CSV file
combined_results = []
for model_results in results.values():
    combined_results.extend(model_results)
combined_results_df = pd.DataFrame(combined_results)
combined_csv_file_path = os.path.join("output", "combined_results.csv")
combined_results_df.to_csv(combined_csv_file_path, index=False)
# print(f"The combined CSV file was created successfully here: {os.path.abspath(combined_csv_file_path)}")

# For saving the best model of each type for each dataset pair
for pair, pair_results in pair_best_model.items():
    for model_name, best_model in pair_results.items():
        model_file_path = os.path.join("models", f"{pair}_{model_name}_best_model.pkl")
        joblib.dump(best_model, model_file_path)
        # print(f"The best model file for {model_name} on {pair} was created successfully here: {os.path.abspath(model_file_path)}")

# Plot ROC curves, PR curves, histograms, and confusion matrices for best models on each dataset pair
for pair_name, X_train_full, X_val_full in dataset_pairs:
    X_val, y_val = X_val_full.drop("CLASS", axis=1), X_val_full["CLASS"]
    roc_fig, roc_ax = plt.subplots()
    pr_fig, pr_ax = plt.subplots()
    cm_fig, cm_ax = plt.subplots(figsize=(8, 7))
    hist_fig, hist_ax = plt.subplots(figsize=(8, 6))

    # Retrieve the best models for the current pair
    best_models = best_results[pair_name]

    # Sort models by AUC-ROC score
    sorted_models = sorted(best_models, key=lambda item: item['auc_roc'], reverse=True)

    for model_data in sorted_models:
        model_name = model_data['model']
        model_file_path = os.path.join("models", f"{pair_name}_{model_name}_best_model.pkl")
        best_model = joblib.load(model_file_path)  # Load the trained model from the file

        # Retrieve the list of feature names used during training
        feature_names = set(best_model.feature_names_in_)

        # Select the corresponding features from X_val
        X_val_selected = X_val[X_val.columns.intersection(feature_names)]

        # Confusion Matrix
        y_pred = best_model.predict(X_val_selected)
        cm = confusion_matrix(y_val, y_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)

        cm_df = pd.DataFrame(cm, index=np.unique(y_val), columns=np.unique(y_val))
        cm_df.index.name = 'Actual'
        cm_df.columns.name = 'Predicted'
        sns.heatmap(cm_df, annot=annot, fmt='', ax=cm_ax, cbar=False, cmap='Blues')
        cm_ax.set_title(f'Confusion Matrix\n{model_name} - {pair_name}')
        cm_file_path = os.path.join("output", f"cm_{pair_name}_{model_name}.svg")
        cm_fig.savefig(cm_file_path, bbox_inches='tight')
        cm_ax.clear()
        # print(f"The confusion matrix SVG file for {model_name} on {pair_name} was outputted successfully here: {os.path.abspath(cm_file_path)}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val_selected)[:, 1])
        roc_auc = round(auc(fpr, tpr) * 100, 2)
        roc_ax.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc}%)')

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val_selected)[:, 1])
        sort_indices = np.argsort(recall)
        precision_sorted = precision[sort_indices]
        recall_sorted = recall[sort_indices]
        pr_auc = round(auc(recall_sorted, precision_sorted) * 100, 2)
        pr_ax.plot(recall_sorted, precision_sorted, label=f'{model_name} (area = {pr_auc}%)')

        # Histogram of Predicted Probabilities
        y_proba = best_model.predict_proba(X_val_selected)[:, 1]
        hist_ax.hist([y_proba[y_val == 0], y_proba[y_val == 1]], bins=10, edgecolor='k',
                     alpha=0.7, color=['blue', 'red'], label=['Predicted 0', 'Predicted 1'])
        hist_ax.set_xlabel('Predicted Probability')
        hist_ax.set_ylabel('Frequency')
        hist_ax.set_title(f'Histogram of Predicted Probabilities\n{model_name} - {pair_name}')
        hist_ax.legend(loc='upper right')
        hist_file_path = os.path.join("output", f"hist_{pair_name}_{model_name}.svg")
        hist_fig.savefig(hist_file_path, bbox_inches='tight')
        hist_ax.clear()
        # print(f"The histogram of predicted probabilities SVG file for {model_name} on {pair_name} was outputted successfully here: {os.path.abspath(hist_file_path)}")

    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title(f'ROC Curve\n{pair_name}')
    roc_ax.legend(loc="lower right")

    roc_file_path = os.path.join("output", f"roc_{pair_name}.svg")
    roc_fig.savefig(roc_file_path, bbox_inches='tight')
    plt.close(roc_fig)
    # print(f"The ROC curve for {pair_name} was saved successfully here: {os.path.abspath(roc_file_path)}")

    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title(f'Precision-Recall Curve\n{pair_name}')
    pr_ax.legend(loc="lower left")

    pr_file_path = os.path.join("output", f"pr_{pair_name}.svg")
    pr_fig.savefig(pr_file_path, bbox_inches='tight')
    plt.close(pr_fig)
    # print(f"The Precision-Recall curve for {pair_name} was saved successfully here: {os.path.abspath(pr_file_path)}")

# Define the model types to iterate over
model_types = ['RF', 'XGB', 'AdaBoost', 'DecisionTree', 'LogReg', 'GBM', 'NaiveBayes', 'MLP']

features = [
    'GROseq',
    'GROseq_RNAseq',
    'GROseq_RNAseq_DNase',
    'GROseq_RNAseq_DNase_H3K4me3',
    'GROseq_RNAseq_DNase_H3K4me3_H3K36me3',
    'all features']

# Splitting and creating dictionary of dataset pairs
dataset_pairs_dict = {}
for cell_combination, X_train, X_test in dataset_pairs:
    if cell_combination.count("_") == 2:
        cell_1 = cell_combination.split("_")[0]
        cell_2 = cell_combination.split("_to_")[1].split("_", 1)[0]
        subset = "all features"
    else:
        cell_1 = cell_combination.split("_")[0]
        cell_2 = cell_combination.split("_to_")[1].split("_", 1)[0]
        subset = cell_combination.split("_", 1)[1].split("_to_", 1)[0]
    dataset_pairs_dict[(cell_1, cell_2, subset)] = {
        "X_train": X_train,
        "X_test": X_test
    }

# Load models with all features
models = {}
for model_type in model_types:
    for file_name in os.listdir("models"):
        if (
            model_type in file_name
            and "_best_model.pkl" in file_name
            and file_name.count("_") == 5
        ):
            cell_1 = file_name.split("_", 1)[0]
            cell_2 = file_name.split("_to_", 1)[1].split("_", 1)[0]
            subset = "all features"
            model = joblib.load(os.path.join("models", file_name))
            models[(cell_1, cell_2, model_type, subset)] = {"model": model, "subset": subset}

# Load models with subsets based on features
for feature in features:
    for model_type in model_types:
        for file_name in os.listdir("models"):
            if (
                model_type in file_name
                and "_best_model.pkl" in file_name
                and f"_{feature}_to" in file_name
            ):
                cell_1 = file_name.split("_", 1)[0]
                cell_2 = file_name.split("_to_", 1)[1].split("_", 1)[0]
                subset = feature
                model = joblib.load(os.path.join("models", file_name))
                models[(cell_1, cell_2, model_type, subset)] = {"model": model, "subset": subset}
                
# Iterate over the model types
for model_type in model_types:
    # Create a separate plot for each cell combination
    for cell_combination in set([(cell_1, cell_2) for (cell_1, cell_2, subset) in dataset_pairs_dict.keys()]):
        fig_roc, ax_roc = plt.subplots()
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"AUC/ROC - {model_type} - {cell_combination[0]} to {cell_combination[1]}")

        fig_pr, ax_pr = plt.subplots()
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"AUC/PR - {model_type} - {cell_combination[0]} to {cell_combination[1]}")

        for feature in features:
            if (cell_combination[0], cell_combination[1], feature) in dataset_pairs_dict:
                data = dataset_pairs_dict[(cell_combination[0], cell_combination[1], feature)]
                X_train = data["X_train"]
                X_test = data["X_test"]
                feature_columns = X_test.columns

                if (cell_combination[0], cell_combination[1], model_type, feature) in models:
                    model = models[(cell_combination[0], cell_combination[1], model_type, feature)]["model"]
                    subset = models[(cell_combination[0], cell_combination[1], model_type, feature)]["subset"]

                    X_data = np.array(X_test[feature_columns])
                    y_test = np.array(X_test["CLASS"])  # Extracting y_test from X_test

                    y_pred = model.predict_proba(X_data[:, :-1])[:, 1]

                    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred)
                    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred)

                    pr_auc = auc(recall, precision) * 100
                    roc_auc = auc(fpr, tpr) * 100

                    ax_pr.plot(recall, precision, label=f"{feature.replace('_', ' ')} (AUC = {pr_auc:.2f}%)")
                    ax_roc.plot(fpr, tpr, label=f'{feature.replace("_", " ")} (AUC = {roc_auc:.2f}%)')

        # Set legend font size and reduce legend size
        legend_fontsize = 10
        ax_pr.legend(handles=ax_pr.get_lines(), labels=[line.get_label() for line in ax_pr.get_lines()], prop={"size": legend_fontsize})
        ax_roc.legend(handles=ax_roc.get_lines(), labels=[line.get_label() for line in ax_roc.get_lines()], prop={"size": legend_fontsize})

        # Adjust the layout to prevent overlapping of labels
        fig_pr.tight_layout()
        fig_roc.tight_layout()
        
         # Save the PR plot
        pr_plot_filename = f"output/pr_best_{cell_combination[0]}_to_{cell_combination[1]}_{model_type}.svg"
        fig_pr.savefig(pr_plot_filename, format="svg")

        # Save the AUC plot
        auc_plot_filename = f"output/auc_best_{cell_combination[0]}_to_{cell_combination[1]}_{model_type}.svg"
        fig_roc.savefig(auc_plot_filename, format="svg")      # Assign label and color based on feature name
