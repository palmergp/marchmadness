import json
import pickle
import os
import yaml
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train(datapath, featurepath, model_set, outpath, model_names):
    """The train function performs the training process based on the input provided
    The split trainer creates two classifier per model. One for early rounds and one for late
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
    """
    # Set outpath
    outpath_full = f"./{outpath}{model_set}"

    # Load featurelist
    with open(featurepath, "r") as f:
        featurenames_short = json.load(f)
    with open(datapath, 'rb') as f:
        all_data = pickle.load(f)

    all_featurenames = ["SeedDiff"]
    for prefix in ["favorite_", "underdog_"]:
        for fname in featurenames_short:
            if fname != "SeedDiff":
                all_featurenames.append(prefix + fname)

    results = []
    models = {}
    # Repeat for both the early and late model
    for type in ["early","late"]:
        if type == "early":
            data = all_data[all_data['round']<3]
        else:
            data = all_data[all_data['round']>=3]
        print(f"Building {type} model")
        # Remove any features not in the featurenames file
        filtered_data = data[data.columns.intersection(all_featurenames)]

        # Structure all data for training
        training_data = filtered_data.values.tolist()
        train_labels = data['favorite_label'].tolist()
        featurenames = list(filtered_data.columns)

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
        print(f"Ranking for {type}:")
        for sf in best_features:
            rank = rank +1
            print("\t" + str(rank) +". "+ featurenames[sf])

        # Scale the features
        scaler = StandardScaler()
        scaled_training_data = scaler.fit_transform(training_data)

        for m in model_names:
            model_package = {}
            print("Starting {}".format(m))
            if m == "Gaussian_Naive_Bayes":
                clf = GaussianNB()
            elif m == "Neural_Network":
                clf = MLPClassifier(random_state=1, max_iter=50000)
            elif m == "Logistic_Regression":
                clf = LogisticRegression()
            elif m == "Linear_SVC":
                clf = svm.SVC(kernel='linear', C=1, probability=True)
            elif m == "KNN":
                clf = KNeighborsClassifier(n_neighbors=10)
            elif m == "Gaussian_RBF":
                kernel = 1.0 * RBF(1.0)
                clf = GaussianProcessClassifier()
            elif m == "Decision_Tree":
                clf = DecisionTreeClassifier(max_depth=5)
            elif m == "Random_Forest":
                clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
            elif m == "Adaboost":
                clf = AdaBoostClassifier()
            else:
                print("Error: Invalid model")

            model = clf.fit(scaled_training_data, train_labels)

            # Get the bg_dist_samp and save it to the package for shap
            model_package["bg_dist_samp"] = pd.DataFrame(scaled_training_data, columns=featurenames)

            model_package["model"] = clf
            model_package["feature_names"] = featurenames
            model_package["scaler"] = scaler
            models[type+"_"+m] = model_package
            scores = cross_val_score(clf, scaled_training_data, train_labels, cv=5, scoring='f1_macro')
            results.append(scores.mean())

    # Add model names for early and late
    early_names = ["early_" + s for s in model_names]
    late_names = ["late_" + s for s in model_names]
    model_names = early_names + late_names
    # Order from least accurate to most
    z = zip(results, model_names)
    sorted_pair = sorted(z)
    tuples = zip(*sorted_pair)
    results, model_names = [list(t) for t in tuples]
    print("\nAccuracies:")
    for i in range(0, len(model_names)):
        print("\t{}:\t {}".format(model_names[i], results[i]))

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
                f.write("{}: {}\n".format(model_names[i], results[i]))


if __name__ == '__main__':
    # Load the config'
    with open("./configs/trainer_config.yml", 'r') as file:
        config = yaml.safe_load(file)
    train(config["data"], config["feature_list"], config["version"], config["outpath"], config["model_names"])
