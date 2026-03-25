"""This file is used to do analysis to determine if there is a significant difference in the feature space from the
first two rounds to the rest"""
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import json
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import ks_2samp


# Load data
print("Loading data")
datapath = 'C:\\Users\\gppal\\PycharmProjects\\marchmadness\\scraping\\data\\training_data.pckl'
with open(datapath, 'rb') as f:
    data = pickle.load(f)
print("Finished loading")

# Modify the data
early = data[data["round"].isin([2])]
late = data[data["round"].isin([3,4,5,6])]

# Get relevant features
feature_path = "C:\\Users\\gppal\\PycharmProjects\\marchmadness\\featuresets\\featuresets25\\v25_2_0\\featurenames_selected.json"
with open(feature_path, "rb") as f:
    feature_names = json.load(f)
# Add favorite and underdog labels
feature_cols = []
for prefix in ["favorite_", "underdog_"]:
    for fname in feature_names:
        if fname != "SeedDiff":
            feature_cols.append(prefix + fname)

# Check distribution shift
results = []
for col in feature_cols:
    stat, p = ks_2samp(early[col], late[col])
    results.append((col, stat, p))

sorted(results, key=lambda x: x[1], reverse=True)
ks_df = pd.DataFrame(results, columns=["feature", "ks_stat", "p_value"])
ks_df = ks_df.sort_values("ks_stat", ascending=False)
print("KS DF")
print(ks_df)

ks = ks_df["ks_stat"]

summary = {
    "mean": ks.mean(),
    "median": ks.median(),
    "p25": ks.quantile(0.25),
    "p75": ks.quantile(0.75),
    "count_gt_0.1": (ks > 0.1).sum(),
    "count_gt_0.2": (ks > 0.2).sum(),
    "count_gt_0.3": (ks > 0.3).sum(),
    "total_features": len(ks)
}
print(summary)

sns.set(style="whitegrid")

# Explicit, stable color mapping
palette = {"expected": "blue", "upset": "orange"}

for col in feature_cols:
    plt.figure(figsize=(12, 5))

    # Early rounds subplot
    plt.subplot(1, 2, 1)
    sns.kdeplot(
        data=early,
        x=col,
        hue="favorite_label",
        palette=palette,
        fill=True,
        common_norm=False,
        alpha=0.5
    )
    plt.title(f"{col} — Rounds 1")

    # Late rounds subplot
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        data=late,
        x=col,
        hue="favorite_label",
        palette=palette,
        fill=True,
        common_norm=False,
        alpha=0.5
    )
    plt.title(f"{col} — Rounds 2")

    plt.suptitle(f"Distribution of {col} by Label and Round Group", fontsize=14)
    plt.tight_layout()
    plt.show()

    # input("Continue?")

print("Done")
