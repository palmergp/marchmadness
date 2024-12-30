"""Using a model, collects statistics on how it would have performed on past tourneys"""
from bracket_predictor import BracketPredictor
from nonsense.favorite_picker import FavoritePicker
import datetime
import numpy as np
import pandas as pd
import os


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
        total_points, picked_winner = predictor.main(True)
        all_totals.append(total_points)

    # Get stats
    min_pts = min(all_totals)
    mean_pts = np.mean(all_totals)
    max_pts = max(all_totals)
    median_pts = np.median(all_totals)
    std_pts = np.std(all_totals)
    stats = {
        "min_pts": min_pts,
        "mean_pts": mean_pts,
        "max_pts": max_pts,
        "median_pts": median_pts,
        "std_pts": std_pts,
        "all_totals": all_totals
    }
    print(f"Results:\n\tMin: {min_pts}\n\tMax: {max_pts}\n\tMean: {mean_pts}\n\tMedian: {median_pts}\n\tStd: {std_pts}")
    print(f"Scores: {all_totals}")
    return stats


def create_bracket_stat_csv(filepath, stats):
    """Saves off the stats of the bracket assessment to a csv"""
    # Write to a CSV
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(stats, orient='index')
    # Reset the index and rename the index column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Models'}, inplace=True)
    # Save the DataFrame to a CSV file with the first column named "Models"
    df.to_csv(filepath + '/bracket_scores.csv', index=False)


if __name__ == '__main__':
    version = "v25_1_0"
    path = f"models/models25/{version}/"
    # Get filenames
    file_names = os.listdir(path)
    file_names = [f for f in file_names if f.endswith("package") and os.path.isfile(os.path.join(path, f))]
    all_stats = {}
    for model in file_names:
        model_stats = collect_bracket_stats(path+model)
        all_stats[model] = model_stats
    create_bracket_stat_csv(path, all_stats)
