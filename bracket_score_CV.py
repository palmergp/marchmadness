"""This will train the model on all years but one, then evaluate the final score. It will then rotate which year is
excluded and rerun until all years have been tested. Finally, it will train on ALL years and output the average score
from the rotations. In theory, the one that did the best across all is the one that should be used"""
import copy
import yaml
from trainer import train
import os
from bracket_predictor import BracketPredictor
import pandas as pd

# Load the config
with open("./configs/trainer_config.yml", 'r') as file:
    config = yaml.safe_load(file)

years = list(range(2011, 2025))
# Loop through each year
count = 0
point_totals = [[] for _ in config["model_names"]] + [[] for _ in config["meta_models"]]
for test_year in years:
    print(f"Test year: {test_year}")
    if test_year == 2020:
        continue
    active_years = [year for year in years if year != test_year]
    updated_version = f"{config['version']}_{count}"
    train(config["data"],
          config["feature_list"],
          updated_version,
          config["outpath"],
          copy.deepcopy(config["model_names"]),
          active_years,
          config["meta_models"],
          config["model_stacks"],
          config["tuning"],
          config["scoring"],
          False,
          False
          )

    # Set outpath
    outpath_full = f"./{config['outpath']}{updated_version}"
    # Get filenames
    file_names = os.listdir(outpath_full)
    file_names = [f for f in file_names if f.endswith("package") and os.path.isfile(os.path.join(outpath_full, f))]
    for idx in range(0, len(file_names)):
        # Create a bracket predictor
        bp = BracketPredictor(outpath_full + "/" + file_names[idx], test_year)
        total_points, picked_winner = bp.main(True)
        point_totals[idx].append(total_points)  # We're trusting that the indices line up every time. Not great

    count += 1

# Create a CSV with the average scores across all
df = pd.DataFrame({'Model': file_names, 'Scores': point_totals})

# Add a column for Average Score
df['Average Score'] = df['Scores'].apply(lambda x: sum(x) / len(x))
df['Median Score'] = df['Scores'].apply(lambda x: pd.Series(x).median())
df['STD Score'] = df['Scores'].apply(lambda x: pd.Series(x).std())
df['Min Score'] = df['Scores'].apply(lambda x: min(x))
df['Max Score'] = df['Scores'].apply(lambda x: max(x))

# Write to CSV
# Set outpath
outpath_full = f"./{config['outpath']}{config['version']}"
# Create the output folder if needed
if not os.path.exists(outpath_full):
    os.makedirs(outpath_full)
df.to_csv(outpath_full + '/bracket_CV_scores.csv', index=False)

# Finally train on all data
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
      False,
      True
      )
