import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from Trainer import extract_features
import numpy as np
import json
import matplotlib.pyplot as plt

def scale_features(feature_file, featurenames):

    with open(feature_file, "rb") as f:
        data = pickle.load(f)

    nonfeatures = ['Year', 'Team1', 'Team2', 'Winner']
    features, featurenames, labels = extract_features(data, featurenames)
    features = np.array(features)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(features)
    return x_train, featurenames, labels


if __name__ == '__main__':
    feature_file = "featuresets/v2_0/feature_data_v2_0.pickle"
    file = open("featuresets/v2_0/featurenames_selected.json", "r")
    featurenames_short = json.load(file)
    featurenames = []
    for fn in featurenames_short:
        featurenames.append("Team1" + fn)
        featurenames.append("Team2" + fn)
    file.close()
    train, featurenames, train_labels = scale_features(feature_file, featurenames)
    # Do feature importance
    chi2_val, p_values = chi2(train, train_labels)
    z = zip(chi2_val, featurenames)
    sorted_pair = sorted(z)
    tuples = zip(*sorted_pair)
    chi2_val, featurenames = [list(t) for t in tuples]
    for i in range(0,len(featurenames)):
        print("{}: {}".format(featurenames[i], chi2_val[i]))
    plt.bar(featurenames, chi2_val)
    plt.title("Feature Chi^2")
    plt.xlabel("Chi^2")
    plt.ylabel("Feature")
    plt.show()