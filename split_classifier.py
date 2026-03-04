from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import pickle


class SplitClassifier(BaseEstimator, ClassifierMixin):
    """The split classifier acts like a normal model but is actually two different models. Which model is used for a
    particular prediction is determined by the round of the game"""
    def __init__(self, early_model, late_model, early_rounds=(1, 2)):
        self.early_model = early_model
        self.late_model = late_model
        self.early_rounds = early_rounds

    def fit(self, X, y):
        # Ensure X is a DataFrame so we can index by column name
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        early_mask = X["round"].isin(self.early_rounds)
        late_mask = ~early_mask

        # Fit each model on its subset
        self.early_model.fit(X.loc[early_mask].drop(columns=["round"]), y.loc[early_mask])
        self.late_model.fit(X.loc[late_mask].drop(columns=["round"]), y.loc[late_mask])

        return self

    def predict(self, X):
        X = pd.DataFrame(X).copy()
        preds = np.empty(len(X), dtype=object)

        early_mask = X["round"].isin(self.early_rounds)
        late_mask = ~early_mask

        preds[early_mask] = self.early_model.predict(X.loc[early_mask].drop(columns=["round"]))
        preds[late_mask] = self.late_model.predict(X.loc[late_mask].drop(columns=["round"]))

        return preds

    def predict_proba(self, X):
        X = pd.DataFrame(X).copy()
        probs = np.zeros((len(X), 2))

        early_mask = X["round"].isin(self.early_rounds)
        late_mask = ~early_mask

        probs[early_mask] = self.early_model.predict_proba(X.loc[early_mask].drop(columns=["round"]))
        probs[late_mask] = self.late_model.predict_proba(X.loc[late_mask].drop(columns=["round"]))

        return probs


def build_split_classifier_pkg(early_model_pkg, late_model_pkg, early_rounds, outpath):
    """Given two model packages, create a new package with a split classifier. Results are saved to provided output
    folder

    Inputs:
        - early_model_pkg: () package of the early model you wish to split with
        - late_model_pkg: () package of the late model you wish to split with
        - early_rounds: ([ints]) ints of rounds to be included with the early model. Valid options are 1-6
        - outpath: (str) path where new package should be saved
    Outputs:
        - split_model_pkg: () package containing split model
    """
    # Load each model package
    package = pickle.load(early_model_pkg)
    early_model = package["model"]
    # From classifier in init
    df = package["bg_dist_samp"]
    f = lambda x: early_model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(f, df)
    features = package["feature_names"]
    if "scaler" in package:
        self.scaler = package["scaler"]
    else:
        self.scaler = None