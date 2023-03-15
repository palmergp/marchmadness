import json
import pickle
import os
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

def train(datapath,featurepath,model_set):

    # Load featurelist
    with open(featurepath,"r") as f:
        featurenames_short = json.load(f)
    with open(datapath,'rb') as f:
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

    model_names = ["Gaussian_Naive_Bayes", "Neural_Network", "Logistic_Regression", "Linear_SVC",
                   "KNN", "Gaussian_RBF", "Decision_Tree", "Random_Forest", "Adaboost"]
    #model_names =["Gaussian_Naive_Bayes"]

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

        model = clf.fit(training_data, train_labels)
        models[m] = clf
        # preds = clf.predict(test)
        # results.append(accuracy_score(test_labels, preds))
        scores = cross_val_score(clf, training_data, train_labels, cv=5, scoring='f1_macro')
        results.append(scores.mean())

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
        outpath = "models/models23/{}".format(model_set)
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
            for i in range(0, len(model_names)):
                f.write("{}: {}\n".format(model_names[i], results[i]))

        # Save feature names
        with open(outpath + "/featurenames.pickle", "wb") as f:
            pickle.dump(featurenames, f)
            f.close()


if __name__ == '__main__':
    feature_list = './featuresets/featuresets23/v23_0_0/featurenames_selected.json'
    data = './scraping/data/training_data.pckl'
    version = 'v23_0_0'
    train(data,feature_list,version)
