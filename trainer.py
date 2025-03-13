import json
import pickle
import os
import yaml
import random
import time
from datetime import datetime
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from model_bracket_stats import collect_bracket_stats, create_bracket_stat_csv

random.seed(11001)


def create_model(model_name, short_p_grid=False):
    """Given a string, a model name and p_grid is returned
    Input:
        - model_name: (str) name of a model to be created
        - short_p_grid: (bool) if raised, the p_grid for hyperparameter tuning will be significantly smaller
    """
    if model_name == "Gaussian_Naive_Bayes":
        clf = GaussianNB()
        p_grid = {}
    elif model_name == "Neural_Network":
        clf = MLPClassifier(max_iter=50000)
        if short_p_grid:
            p_grid = {'solver': ['sgd'],
                      'max_iter': [1000],
                      'learning_rate_init': [0.1],
                      'learning_rate': ['adaptive'],
                      'hidden_layer_sizes': [50,],
                      'early_stopping': [False],
                      'alpha': [0.001],
                      'activation': ['tanh']
                      }
        else:
            p_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs', 'sgd', 'adam'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      'learning_rate_init': [0.001, 0.01, 0.1],
                      'max_iter': [200, 500, 1000],
                      'early_stopping': [True, False]}
    elif model_name == "Logistic_Regression":
        clf = LogisticRegression()
        if short_p_grid:
            p_grid = {'solver': ['lbfgs'],
                      'penalty': ['l2'],
                      'max_iter': [500],
                      'l1_ratio': [0.1],
                      'C': [0.001]
                      }
        else:
            p_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'max_iter': [100, 200, 500],
                      'l1_ratio': [0.1, 0.5, 0.9]}
    elif model_name == "Linear_SVC":
        clf = svm.SVC(kernel='linear', probability=True)
        if short_p_grid:
            p_grid = {'C': [0.01, 0.1, 1, 10, 100],
                      'max_iter': [1000, 2000, 5000],
                      'tol': [0.0001, 0.001, 0.01, 0.1],
                      'class_weight': [None, 'balanced']}
        else:
            p_grid = {'C': [0.01, 0.1, 1, 10, 100],
                      'max_iter': [1000, 2000, 5000],
                      'tol': [0.0001, 0.001, 0.01, 0.1],
                      'class_weight': [None, 'balanced']}
    elif model_name == "KNN":
        clf = KNeighborsClassifier()
        if short_p_grid:
            p_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'minkowski'],
                      'p': [1, 2]}
        else:
            p_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'minkowski'],
                      'p': [1, 2]}
    elif model_name == "Gaussian_RBF":
        clf = GaussianProcessClassifier()
        if short_p_grid:
            p_grid = {'n_restarts_optimizer': [0, 5, 10],
                      'max_iter_predict': [100, 200],
                      'warm_start': [False, True]}
        else:
            p_grid = {'n_restarts_optimizer': [0, 5, 10],
                      'max_iter_predict': [100, 200],
                      'warm_start': [False, True]}
    elif model_name == "Decision_Tree":
        clf = DecisionTreeClassifier()
        if short_p_grid:
            p_grid = {'max_depth': [None, 10, 20, 30, 40, 50],
                      'min_samples_split': [2, 5, 10, 20],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'max_features': [None, 'auto', 'sqrt', 'log2'],
                      'criterion': ['gini', 'entropy']}
        else:
            p_grid = {'max_depth': [None, 10, 20, 30, 40, 50],
                      'min_samples_split': [2, 5, 10, 20],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'max_features': [None, 'auto', 'sqrt', 'log2'],
                      'criterion': ['gini', 'entropy']}
    elif model_name == "Random_Forest":
        clf = RandomForestClassifier()
        if short_p_grid:
            p_grid = {'n_estimators': [50, 300, 500],
                      'max_features': ['auto'],
                      'max_depth': [None, 20, 50],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False]}
        else:
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [None, 10, 20, 30, 40, 50],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False]}
    elif model_name == "Adaboost":
        clf = AdaBoostClassifier()
        if short_p_grid:
            p_grid = {'n_estimators': [50],
                      'learning_rate': [0.01],
                      'base_estimator': [DecisionTreeClassifier(max_depth=2)]}
        else:
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'learning_rate': [0.01, 0.1, 0.5, 1, 1.5, 2],
                      'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2),
                                         DecisionTreeClassifier(max_depth=3)]}
    elif model_name == "KernelSVM":
        clf = svm.SVC(probability=True)
        if short_p_grid:
            p_grid = {'kernel': ['linear', 'poly', 'rbf'],
                      'C': [0.1, 1, 10],
                      'gamma': [0.001, 0.01, 0.1]}
        else:
            p_grid = {'kernel': ['linear', 'poly', 'rbf'],
                      'C': [0.1, 1, 10],
                      'gamma': [0.001, 0.01, 0.1]}
    elif model_name == "GradientBoost":
        clf = GradientBoostingClassifier()
        if short_p_grid:
            p_grid = {}
        else:
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                      'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                      'max_features': [None, 'auto', 'sqrt', 'log2']}
    else:
        print(f"Error: Invalid model - {model_name}")
        clf = None
        p_grid = None

    return clf, p_grid


def train(datapath, featurepath, model_set, outpath, model_names, training_years=[],
          meta_models=[], model_stacks=[], tuning=True, scoring="accuracy", feature_analysis=False, bracket_stats=True):
    """The train function performs the training process based on the input provided
    Input:
        - datapath: (str) path to the data file containing all feature data
        - featurepath: (str) path to file detailing the full list of features to use when training
        - model_set: (str) a tag indicating the model version. Used when making the output folder
        - outpath: (str) path to where the output data should be saved. A new folder will be created in this location
        - model_names: (str) list of algorithms to be used. A separate model will be created for each. Must be one of
                        the following options:
                        - Gaussian_Naive_Bayes
                        - Neural_Network
                        - Logistic_Regression
                        - Linear_SVC
                        - KNN
                        - Gaussian_RBF
                        - Decision_Tree
                        - Random_Forest
                        - Adaboost
                        - GradientBoost
                        - KernelSVM
        - training_years: (list) list of years to include in training
        - tuning: (bool) flag indicating whether hyperparameter tuning should be done or not,
        - scoring: (str) Indicates what scoring method should be used for hypertuning. Options:
            - accuracy
            - roc_auc
        - feature analysis: (bool) flag indicating whether to do feature analysis
        - bracket_stats: (bool) flag indicating whether scores should be calculated for each year in training
    """
    start_time = time.time()
    # Set outpath
    outpath_full = f"./{outpath}{model_set}"

    # Create the output folder if needed
    if not os.path.exists(outpath_full):
        os.makedirs(outpath_full)

    # Load featurelist
    with open(featurepath, "r") as f:
        featurenames_short = json.load(f)
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    all_featurenames = ["SeedDiff"]
    for prefix in ["favorite_", "underdog_"]:
        for fname in featurenames_short:
            if fname != "SeedDiff":
                all_featurenames.append(prefix + fname)

    # Remove any features not in the featurenames file
    train_data = data[data['year'].isin(training_years)]
    filtered_data = train_data[train_data.columns.intersection(all_featurenames)]

    # Change labels from strings to ints
    train_data["favorite_label"] = train_data["favorite_label"].map({'expected': 0, 'upset': 1})

    # Structure all data for training
    train_labels = train_data['favorite_label'].tolist()
    training_data = filtered_data.values.tolist()
    featurenames = list(filtered_data.columns)

    if feature_analysis:
        # Create the correlation heatmap for features
        corr_matrix = filtered_data.corr().abs()
        # sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
        # Unstack the matrix to get a dataframe of correlations
        corr_pairs = corr_matrix.unstack()
        # Remove self-correlations
        corr_pairs = corr_pairs[corr_pairs != 1]
        # Sort the pairs by correlation value
        sorted_pairs = corr_pairs.sort_values(ascending=False)
        # Display the top correlations
        print("Highest Correlated Features:")
        print(sorted_pairs.head(10))

        # Create and fit the selector object
        selector = SelectKBest(f_classif, k='all')
        selector.fit(training_data, train_labels)
        # Get the selected features
        selected_features = selector.get_support(indices=True)
        # Get the scores of each feature
        scores = selector.scores_

        # Sort the scores in descending order and get the indices
        sorted_indices = np.argsort(scores)[::-1]

        # Get the 10 best features in order of importance
        best_features = sorted_indices[:]
        rank = 0
        print("Top 10 most important features:")
        for sf in best_features:
            rank = rank + 1
            print("\t" + str(rank) + ". " + featurenames[sf])

    # Scale the features
    scaler = StandardScaler()
    scaled_training_data = scaler.fit_transform(training_data)

    # Split training and test 80/20
    X_train, X_test, y_train, y_test = train_test_split(scaled_training_data, train_labels, test_size=0.20,
                                                        random_state=42)

    results = []
    params = {}
    models = {}
    # Train the single models
    for m in model_names:
        model_package = {}
        print(f"{datetime.now()}: Starting {m}")
        clf, p_grid = create_model(m)

        if tuning and p_grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                param_search = RandomizedSearchCV(clf, p_grid, cv=5, scoring=scoring)
                param_search.fit(X_train, y_train)

            # Store the best model and its performance
            clf = param_search.best_estimator_
            print(f"Best Parameters: {param_search.best_params_}")
            params[m] = param_search.best_params_
        else:
            model = clf.fit(X_train, y_train)
            params[m] = "Default"

        # Get the bg_dist_samp and save it to the package for shap
        model_package["bg_dist_samp"] = pd.DataFrame(scaled_training_data, columns=featurenames)

        model_package["model"] = clf
        model_package["feature_names"] = featurenames
        model_package["scaler"] = scaler
        models[m] = model_package
        scores = cross_val_score(clf, scaled_training_data, train_labels, cv=5, scoring='f1_macro')
        results.append(scores.mean())

    # Train stacked models
    for m in meta_models:
        base_models = []
        for stack in model_stacks:
            stack = stack.lower()
            # The all stack uses every model we've created this time around
            if stack == "all":
                base_models = [(m, models[m]["model"]) for m in models]
            # The top5_accuracy uses 5 models with the highest accuracy
            elif stack == "top5_accuracy":
                print("TOP5 ACCURACY NOT IMPLEMENTED")
            # The top5_roc uses 5 models with the highest ROC AUC
            elif stack == "top5_auc":
                print("TOP5_AUC NOT IMPLEMENTED")

            # If a valid base model configuration was chosen, train the stacking model
            if base_models:
                # Make the name of the model
                stack_name = f"stack_{stack}_{m}"
                # Add the name to the model names list
                model_names.append(stack_name)
                print(f"{datetime.now()}: Starting {stack_name}")
                # Create the meta model
                clf, p_grid = create_model(m, True)
                clf = StackingClassifier(estimators=base_models, final_estimator=clf, cv=5)
                # Perform hyperparameter tuning if flag is raised
                with warnings.catch_warnings():  # When tuning, lots of warnings happen so we want to suppress them
                    warnings.simplefilter("ignore")
                    if tuning and p_grid:
                        # Update p_grid to work with stacking model
                        p_grid = {'final_estimator__' + key: value for key, value in p_grid.items()}
                        # Begin tuning
                        param_search = RandomizedSearchCV(clf, p_grid, cv=5, scoring=scoring)
                        param_search.fit(X_train, y_train)

                        # Store the best model and its performance
                        clf = param_search.best_estimator_
                        print(f"Best Parameters: {param_search.best_params_}")
                        params[stack_name] = param_search.best_params_
                    else:
                        clf.fit(X_train, y_train)
                        params[stack_name] = "Default"

                # Create the model package
                model_package = {"bg_dist_samp": pd.DataFrame(scaled_training_data, columns=featurenames), "model": clf,
                                 "feature_names": featurenames, "scaler": scaler}
                models[stack_name] = model_package
                scores = cross_val_score(clf, scaled_training_data, train_labels, cv=5, scoring='f1_macro')
                results.append(scores.mean())

    # Plot ROC Curves
    if "upset" in y_test:
        y_test = [1 if x == "upset" else 0 for x in y_test]
    auc = {}
    acc = {}
    plt.clf()  # clear figure
    for m in models:
        try:
            yHat = models[m]["model"].predict_proba(X_test)
            roc_pfa, roc_pd, thresh = roc_curve(y_test, yHat[:, 1], pos_label=1)
            plt.plot(roc_pfa, roc_pd, label=m)
            classes = [1 if x >= 0.5 else 0 for x in yHat[:, 1]]
            test_accuracy = accuracy_score(y_test, classes)
            acc[m] = test_accuracy
            roc_auc = roc_auc_score(y_test, yHat[:, 1])
            auc[m] = roc_auc
        except ValueError:
            acc[m] = -1
            auc[m] = -1
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{outpath_full}/ROC.png")

    # Order from least accurate to most
    z = zip(results, model_names)
    sorted_pair = sorted(z)
    tuples = zip(*sorted_pair)
    results, model_names = [list(t) for t in tuples]
    print("\nAccuracies:")
    for i in range(0, len(model_names)):
        print(f"\t{model_names[i]} - CV Accuracy: {results[i]}, Test Accuracy: {acc[model_names[i]]}, Test ROC: {auc[model_names[i]]}")

    # Save off models
    if model_set:
        try:
            os.mkdir(outpath_full)
        except FileExistsError:
            pass

        # save models
        for m in model_names:
            with open(outpath_full + "/" + m + "_" + model_set + ".package", 'wb') as f:
                pickle.dump(models[m], f)
                f.close()

        # Save accuracies
        with open(outpath_full + "/accuracy.txt", "w") as f:
            for i in range(0, len(model_names)):
                #f.write("{}: {}\n".format(model_names[i], results[i]))
                f.write(f"\t{model_names[i]} - Accuracy: {results[i]}, Test Accuracy: {acc[model_names[i]]}, Test ROC: {auc[model_names[i]]}\n")
                f.write(f"\t\tParams: {params[model_names[i]]}\n")

        # Save config
        with open(outpath_full + "/config_params.txt", "w") as f:
            f.write(f"feature list: {featurepath}\n")
            f.write(f"tuning: {tuning}\n")
            f.write(f"scoring: {scoring}\n")
            f.write(f"models: {model_names}\n")
            f.write(f"meta models: {meta_models}\n")
            f.write(f"training_years: {training_years}\n")

    # Calculate bracket stats
    if bracket_stats:
        # Get filenames
        file_names = os.listdir(outpath_full)
        file_names = [f for f in file_names if f.endswith("package") and os.path.isfile(os.path.join(outpath_full, f))]
        all_stats = {}
        # Calculate scores for each model
        for model in file_names:
            model_stats = collect_bracket_stats(outpath_full + "/" + model)
            all_stats[model] = model_stats
        # Save it to a CSV
        create_bracket_stat_csv(outpath_full, all_stats)

    print(f'Total Train Time: {time.time()-start_time}s')


if __name__ == '__main__':
    # Load the config'
    with open("./configs/trainer_config.yml", 'r') as file:
        config = yaml.safe_load(file)
    train(config["data"],
          config["feature_list"],
          config["version"],
          config["outpath"],
          config["model_names"],
          config["training_years"],
          config["meta_models"],
          config["model_stacks"],
          config["tuning"],
          config["scoring"],
          config["feature_analysis"],
          config["bracket_scores"]
          )
