"""Using a model, collects statistics on how it would have performed on past tourneys"""
from bracket_predictor import BracketPredictor
from nonsense.favorite_picker import FavoritePicker
import datetime
import numpy as np

def collect_bracket_stats(model):
    """Goes through every year"""
    predictor = BracketPredictor(model, 2011)
    today = datetime.datetime.now()
    if today.month > 4:
        # If it's after april, then the tourney has finished
        # You can count this year
        max_year = today.year + 1
    else:
        max_year = today.year

    # Go through each year
    all_totals = []
    for year in range(2011, max_year):
        if year == 2020:
            # COVID...
            continue
        predictor.set_year(year)
        total_points = predictor.main(True)
        all_totals.append(total_points)

    # Get stats
    min_pts = min(all_totals)
    mean_pts = np.mean(all_totals)
    max_pts = max(all_totals)
    median_pts = np.median(all_totals)
    std_pts = np.std(all_totals)
    print(f"Results:\n\tMin: {min_pts}\n\tMax: {max_pts}\n\tMean: {mean_pts}\n\tMedian: {median_pts}\n\tStd: {std_pts}")
    print(f"Scores: {all_totals}")


if __name__ == '__main__':
    version = "v25_0_3"
    path = f"models/models25/{version}/"
    #path = "nonsense/"
    model_pkg = f"Gaussian_RBF_{version}.package"
    #model_pkg = "fav_picker.package"
    collect_bracket_stats(path+model_pkg)
