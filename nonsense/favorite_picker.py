import pickle

import pandas as pd


class FavoritePicker:
    """This is a fake model that picks the favorite everytime. This is used as a point of reference to see if the created
    models are any better than the most basic method for making a bracket"""

    def __init__(self):
        pass

    def predict_proba(self, features):
        """Return 100% confidence in the favorite every time"""
        return [[1, 0]]


if __name__ == '__main__':
    # Create a favorite picker and save it as a pickle file
    fav_picker = FavoritePicker()
    model_package = {
        "model": fav_picker,
        "bg_dist_samp": pd.DataFrame(),
        "feature_names": ["favorite_seed","underdog_seed"],
        "scaler": None
    }
    with open("C:\\Users\\gppal\\PycharmProjects\\marchmadness\\nonsense\\fav_picker.package", "wb") as f:
        pickle.dump(model_package, f)
