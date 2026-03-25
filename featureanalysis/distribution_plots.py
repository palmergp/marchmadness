import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ks_2samp
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_separation_metrics(df, feature, label_col):
    x0 = df[df[label_col] == "expected"][feature].dropna()
    x1 = df[df[label_col] == "upset"][feature].dropna()

    # KS statistic
    ks = ks_2samp(x0, x1).statistic

    # Cohen's d
    pooled_std = np.sqrt(((x0.std()**2) + (x1.std()**2)) / 2)
    d = (x1.mean() - x0.mean()) / pooled_std if pooled_std > 0 else 0

    # AUC (univariate)
    try:
        auc = roc_auc_score(df[label_col], df[feature])
    except:
        auc = np.nan

    return ks, d, auc


def compute_metrics(df, feature, label_col):
    x0 = df[df[label_col] == "expected"][feature].dropna()
    x1 = df[df[label_col] == "upset"][feature].dropna()

    # KS statistic
    ks = ks_2samp(x0, x1).statistic

    # Cohen's d
    pooled_std = np.sqrt(((x0.std()**2) + (x1.std()**2)) / 2)
    d = (x1.mean() - x0.mean()) / pooled_std if pooled_std > 0 else 0

    # AUC
    try:
        auc = roc_auc_score(df[label_col], df[feature])
    except:
        auc = np.nan

    return ks, d, auc



def find_high_signal_features(df, feature_list, label_col="favorite_label"):
    results = []

    for feature in feature_list:
        if feature not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        ks, d, auc = compute_metrics(df, feature, label_col)

        # High-signal rule
        high_signal = (
                  int(ks > 0.20) +
                  int(abs(d) > 0.30) +
                  int(auc > 0.60)
          ) >= 2

        results.append({
            "feature": feature,
            "KS": ks,
            "Cohen_d": d,
            "AUC": auc,
            "high_signal": high_signal
        })

    return pd.DataFrame(results).sort_values("KS", ascending=False)


# Load data
with open("C:\\Users\\gppal\\PycharmProjects\\marchmadness\\scraping\\data\\training_data.pckl", "rb") as f:
    df = pickle.load(f)
df = df.reset_index(drop=True)

# ---------------------------------------------------------
# LOAD FEATURE SCHEMA
# ---------------------------------------------------------

with open("C:\\Users\\gppal\\PycharmProjects\\marchmadness\\scraping\\data\\full_featurenames.json", "r") as f:
    feature_schema = json.load(f)

difference_features = feature_schema["difference_stats"]
synergy_features = feature_schema["synergy_stats"]
individual_features = feature_schema["individual_stats"]
# Create prefixed versions
favorite_features = [f"favorite_{feat}" for feat in individual_features]
underdog_features = [f"underdog_{feat}" for feat in individual_features]

# Combine them
individual_prefixed = favorite_features + underdog_features

# Optional: filter to only those that exist in df
individual_prefixed = [feat for feat in individual_prefixed if feat in df.columns]



# Column containing the class label
label_col = "favorite_label"  # adjust if needed

# ---------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------

output_dir = "feature_distributions"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# COLOR MAP FOR CLASSES
# ---------------------------------------------------------

class_colors = {
    "expected": "#1f77b4",  # expected
    "upset": "#ff7f0e"  # upset
}


# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------

def plot_feature_distribution(df, feature, label_col, group_name):
    if feature not in df.columns:
        print(f"Skipping {feature}: not found in dataframe")
        return

    if not pd.api.types.is_numeric_dtype(df[feature]):
        print(f"Skipping {feature}: not numeric")
        return

    plt.figure(figsize=(8, 5))

    for label_value, color in class_colors.items():
        subset = df[df[label_col] == label_value]
        sns.histplot(
            subset[feature],
            kde=False,
            stat="density",
            bins=30,
            color=color,
            label=f"{label_col} = {label_value}",
            alpha=0.5
        )

    ks, d, auc = compute_separation_metrics(df, feature, label_col)

    plt.title(f"{feature}\nKS={ks:.3f}, d={d:.3f}, AUC={auc:.3f}")

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{group_name}_{feature}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# LOOP THROUGH BOTH FEATURE GROUPS
# ---------------------------------------------------------

# for feature in difference_features:
#     plot_feature_distribution(df, feature, label_col, "difference")
#
# for feature in synergy_features:
#     plot_feature_distribution(df, feature, label_col, "synergy")

df_diff = find_high_signal_features(df, difference_features)
df_syn = find_high_signal_features(df, synergy_features)
df_ind = find_high_signal_features(df, individual_prefixed)

high_signal_diff = df_diff[df_diff["high_signal"] == True]
high_signal_syn = df_syn[df_syn["high_signal"] == True]
high_signal_ind = df_ind[df_ind["high_signal"] == True]

# High signal synergy
print("Synergy Features:")
print(high_signal_syn)

print("Diff Features:")
print(high_signal_diff)

print("Individual Features:")
print(high_signal_ind)