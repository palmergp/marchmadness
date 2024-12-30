import json
import pickle
import os
import yaml
import random
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from model_bracket_stats import collect_bracket_stats, create_bracket_stat_csv

random.seed(11001)


def train(datapath, featurepath, model_set, outpath, model_names, tuning, feature_analysis):
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
        - tuning: (bool) flag indicating whether hyperparameter tuning should be done or not
        - feature analysis: (bool) flag indicating whether to do feature analysis
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
    filtered_data = data[data.columns.intersection(all_featurenames)]

    # Structure all data for training
    training_data = filtered_data.values.tolist()
    train_labels = data['favorite_label'].tolist()
    featurenames = list(filtered_data.columns)

    if feature_analysis:
        # Create the correlation heatmap for features
        corr_matrix = filtered_data.corr().abs()
        #sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True)
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
    X_train, X_test, y_train, y_test = train_test_split(scaled_training_data, train_labels, test_size=0.2,
                                                        random_state=42)
    results = []
    params = {}
    models = {}
    for m in model_names:
        model_package = {}
        print("Starting {}".format(m))
        if m == "Gaussian_Naive_Bayes":
            clf = GaussianNB()
            p_grid = {}
        elif m == "Neural_Network":
            clf = MLPClassifier(max_iter=50000)
            p_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs', 'sgd', 'adam'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      'learning_rate_init': [0.001, 0.01, 0.1],
                      'max_iter': [200, 500, 1000],
                      'early_stopping': [True, False]}
        elif m == "Logistic_Regression":
            clf = LogisticRegression()
            p_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'max_iter': [100, 200, 500],
                      'l1_ratio': [0.1, 0.5, 0.9]}
        elif m == "Linear_SVC":
            clf = svm.SVC(kernel='linear', probability=True)
            p_grid = {'C': [0.01, 0.1, 1, 10, 100],
                      'max_iter': [1000, 2000, 5000],
                      'tol': [0.0001, 0.001, 0.01, 0.1],
                      'class_weight': [None, 'balanced']}
        elif m == "KNN":
            clf = KNeighborsClassifier()
            p_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'minkowski'],
                      'p': [1, 2]}
        elif m == "Gaussian_RBF":
            clf = GaussianProcessClassifier()
            p_grid = {'n_restarts_optimizer': [0, 5, 10],
                      'max_iter_predict': [100, 200],
                      'warm_start': [False, True]}
        elif m == "Decision_Tree":
            clf = DecisionTreeClassifier()
            p_grid = {'max_depth': [None, 10, 20, 30, 40, 50],
                      'min_samples_split': [2, 5, 10, 20],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'max_features': [None, 'auto', 'sqrt', 'log2'],
                      'criterion': ['gini', 'entropy']}
        elif m == "Random_Forest":
            clf = RandomForestClassifier()
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [None, 10, 20, 30, 40, 50],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False]}
        elif m == "Adaboost":
            clf = AdaBoostClassifier()
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'learning_rate': [0.01, 0.1, 0.5, 1, 1.5, 2],
                      'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)]}
        elif m == "KernelSVM":
            clf = svm.SVC(probability=True)
            p_grid = {'kernel': ['linear', 'poly', 'rbf'],
                      'C': [0.1, 1, 10],
                      'gamma': [0.001, 0.01, 0.1]}
        elif m == "GradientBoost":
            clf = GradientBoostingClassifier()
            p_grid = {'n_estimators': [50, 100, 200, 300, 400, 500],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                      'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                      'max_features': [None, 'auto', 'sqrt', 'log2']}
        else:
            print("Error: Invalid model")
            continue

        # model = clf.fit(scaled_training_data, train_labels)
        if tuning and p_grid:
            param_search = RandomizedSearchCV(clf, p_grid, cv=5, scoring='accuracy')
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

    # Plot ROC Curves
    y_test = [1 if x == "upset" else 0 for x in y_test]
    auc = {}
    acc = {}
    for m in models:
        yHat = models[m]["model"].predict_proba(X_test)
        roc_pfa, roc_pd, thresh = roc_curve(y_test, yHat[:, 1], pos_label=1)
        plt.plot(roc_pfa, roc_pd, label=m)
        classes = [1 if x >= 0.5 else 0 for x in yHat[:, 1]]
        test_accuracy = accuracy_score(y_test, classes)
        acc[m] = test_accuracy
        roc_auc = roc_auc_score(y_test, yHat[:, 1])
        auc[m] = roc_auc
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

    # Calculate bracket stats
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
          config["tuning"],
          config["feature_analysis"]
          )
