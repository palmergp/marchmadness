import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import os
import json

def extract_features(data, featurenames):
    labels = list(data["Winner"])
    # featurenames = [x for x in list(data.columns) if x not in nonfeatures]
    features = []
    for i in range(0, len(data)):
        row = []
        for fn in featurenames:
            value = data.iloc[i][fn]
            if value == -1:
                print("{} Never updated".format(fn))
            row.append(data.iloc[i][fn])
        features.append(row)
    return features, featurenames, labels


def train(datapath, featurename_path, model_set=None):

    # Load data
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    with open(featurename_path, "r") as f:
        featurenames_short = json.load(f)
    f.close()
    print("The following features will be used for each team:\n{}".format(featurenames_short))
    featurenames = []
    for n in featurenames_short:
        if n == "SeedDiff":
            featurenames.append(n)
        else:
            featurenames.append("Team1"+n)
            featurenames.append("Team2"+n)
    label_names = ["Team1", "Team2"]
    features, featurenames, labels = extract_features(data, featurenames)

    # Split our data
    # train, test, train_labels, test_labels = train_test_split(features,
    #                                                           labels,
    #                                                           test_size=0.3,
    #                                                           random_state=42)
    train = features
    train_labels = labels
    model_names = ["Gaussian_Naive_Bayes", "Neural_Network", "Logistic_Regression", "Linear_SVC",
              "KNN", "Gaussian_RBF", "Decision_Tree", "Random_Forest", "Adaboost"]


    results = []
    models = {}
    for m in model_names:
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

        model = clf.fit(train, train_labels)
        models[m] = clf
        # preds = clf.predict(test)
        # results.append(accuracy_score(test_labels, preds))
        scores = cross_val_score(clf, train, train_labels, cv=5, scoring='f1_macro')
        results.append(scores.mean())

    # Order from least accurate to most
    z = zip(results, model_names)
    sorted_pair = sorted(z)
    tuples = zip(*sorted_pair)
    results, model_names = [list(t) for t in tuples]
    print("\nAccuracies:")
    for i in range(0,len(model_names)):
        print("\t{}:\t {}".format(model_names[i], results[i]))

    # Save off models
    if model_set:
        outpath = "models/{}".format(model_set)
        try:
            os.mkdir(outpath)
        except FileExistsError:
            pass

        # save models
        for m in model_names:
            with open(outpath + "/" + m + "_" + model_set + ".pickle", 'wb') as f:
                pickle.dump(models[m], f)
                f.close()

        # Save accuracies
        with open(outpath + "/accuracy.txt", "w") as f:
            for i in range(0,len(model_names)):
                f.write("{}: {}\n".format(model_names[i], results[i]))

        # Save feature names
        with open(outpath + "/featurenames.pickle", "wb") as f:
            pickle.dump(featurenames, f)
            f.close()


if __name__ == '__main__':
    dpath = "./featuresets/featuresets22/v4_0/feature_data_v4_0.pickle"
    model_set = "v4_1"
    featurename_path = "./featuresets/featuresets22/v4_0/featurenames_selected.json"
    train(dpath, featurename_path, model_set)
