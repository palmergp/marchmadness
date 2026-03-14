import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import shap
import pickle
import warnings
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

bad_features = [
    "games",
    "wins",
    "losses",
    "points",
    "opp_points",
    "minutes_played",
    "field_goals",
    "field_goal_attempts",
    "field_goal_percentage",
    "three_point_field_goals",
    "three_point_field_goal_attempts",
    "free_throws",
    "free_throw_attempts",
    "offensive_rebounds",
    "total_rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "personal_fouls",
    "opp_minutes_played",
    "opp_field_goals",
    "opp_field_goal_attempts",
    "opp_three_point_field_goals",
    "opp_three_point_field_goal_attempts",
    "opp_free_throws",
    "opp_free_throw_attempts",
    "opp_total_rebounds",
    "opp_assists",
    "opp_steals",
    "opp_blocks",
    "opp_turnovers",
    "opp_personal_fouls",
    "defensive_rebounds",
    "two_point_field_goals",
    "two_point_field_goal_attempts",
    "opp_defensive_rebounds",
    "opp_two_point_field_goals",
    "opp_two_point_field_goal_attempts",
    "games_played"
]
bad_features_diff = [
    "points_diff",
    "field_goals_diff",
    "opp_minutes_played_diff",
    "minutes_played_diff",
    "games_diff",
    "wins_diff",
    "losses_diff",
    "opp_points_diff",
    "field_goals_diff",
    "field_goal_attempts_diff",
    "three_point_field_goals_diff",
    "three_point_field_goal_attempts_diff",
    "free_throws_diff",
    "free_throw_attempts_diff",
    "offensive_rebounds_diff",
    "total_rebounds_diff",
    "assists_diff",
    "steals_diff",
    "blocks_diff",
    "turnovers_diff",
    "personal_fouls_diff",
    "opp_field_goals_diff",
    "opp_field_goal_attempts_diff",
    "opp_three_point_field_goals_diff",
    "opp_three_point_field_goal_attempts_diff",
    "opp_free_throws_diff",
    "opp_free_throw_attempts_diff",
    "opp_offensive_rebounds_diff",
    "opp_total_rebounds_diff",
    "opp_assists_diff",
    "opp_steals_diff",
    "opp_blocks_diff",
    "opp_turnovers_diff",
    "opp_personal_fouls_diff",
    "defensive_rebounds_diff",
    "two_point_field_goals_diff",
    "two_point_field_goal_attempts_diff",
    "opp_defensive_rebounds_diff",
    "opp_two_point_field_goals_diff",
    "opp_two_point_field_goal_attempts_diff",
    "games_played_diff"
]

def load_training_data(path, label_col, season_col=None):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    y = df[label_col]
    # Remove unneeded columns
    bad_features_full = []
    for feat in bad_features:
        bad_features_full.append("underdog_" + feat)
        bad_features_full.append("favorite_" + feat)
    X = df.drop(columns=bad_features_full + bad_features_diff + [label_col, "underdog_label"])
    seasons = df[season_col] if season_col else None
    return X, y, seasons


def correlation_pruning(X, threshold=0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_pruned = X.drop(columns=to_drop)
    return X_pruned, to_drop, corr


def compute_mutual_info(X, y):
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series


def compute_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
    perm = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    return perm


def compute_permutation_importance_heldout(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    result = permutation_importance(
        rf,
        X_val,
        y_val,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc"
    )

    perm = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    return perm, rf


def compute_tree_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp, rf

def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def yearly_stability(df, feature_cols, label_col, season_col):
    years = sorted(df[season_col].unique())
    stability = {f: [] for f in feature_cols}

    for i in range(len(years) - 1):
        train_years = years[:i+1]
        test_year = years[i+1]

        train_df = df[df[season_col].isin(train_years)]
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]

        rf = RandomForestClassifier(n_estimators=400, random_state=42)
        rf.fit(X_train, y_train)

        imp = pd.Series(rf.feature_importances_, index=feature_cols)
        for f in feature_cols:
            stability[f].append(imp[f])

    stab_df = pd.DataFrame(stability).T
    stab_df["mean"] = stab_df.mean(axis=1)
    stab_df["std"] = stab_df.std(axis=1)
    stab_df["stability_score"] = stab_df["mean"] / (stab_df["std"] + 1e-6)
    return stab_df.sort_values("stability_score", ascending=False)


def run_feature_analysis(path, label_col, season_col=None, outdir="feature_analysis_output"):
    X, y, seasons = load_training_data(path, label_col, season_col)
    with open(path, 'rb') as f:
        df = pickle.load(f)

    ensure_dir(outdir)
    print("Loaded data")

    # 1. Correlation pruning
    X_corr, dropped_corr, corr_matrix = correlation_pruning(X)
    save_corr_heatmap(corr_matrix, f"{outdir}/correlation/corr_matrix.png")
    save_json(dropped_corr, f"{outdir}/correlation/dropped_correlated.json")
    print("Finished correlation pruning")

    # 2. Mutual information
    mi = compute_mutual_info(X_corr, y)
    save_series(mi, f"{outdir}/mutual_info/mutual_info.csv",
                f"{outdir}/mutual_info/mutual_info.png",
                title="Mutual Information (Top 40)")
    print("Finished mutual information")

    # 3. Tree importance + SHAP
    tree_imp, rf_model = compute_tree_importance(X_corr, y)
    save_series(tree_imp, f"{outdir}/tree_importance/rf_importance.csv",
                f"{outdir}/tree_importance/rf_importance.png",
                title="Random Forest Feature Importance (Top 40)")

    shap_values = compute_shap_values(rf_model, X_corr)
    save_shap_summary(shap_values, X_corr, f"{outdir}/shap/shap_summary.png")
    save_shap_bar(shap_values, X_corr, f"{outdir}/shap/shap_bar.png")
    print("Finished tree importance and SHAP")

    # 4. Permutation importance
    perm_imp, rf = compute_permutation_importance_heldout(X_corr, y)
    save_series(perm_imp, f"{outdir}/permutation_importance_heldout/perm_importance.csv",
                f"{outdir}/permutation_importance_heldout/perm_importance.png",
                title="Permutation Importance (Top 40)")
    print("Finished permutation importnace")

    # 5. Stability (optional)
    stability = None
    if season_col:
        stability = yearly_stability(df, X_corr.columns, label_col, season_col)
        save_dataframe(stability, f"{outdir}/stability/stability_scores.csv")

        plt.figure(figsize=(10, 6))
        stability["stability_score"].sort_values(ascending=False).head(40).plot(kind="bar")
        plt.title("Stability Scores (Top 40)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/stability/stability_scores.png")
        plt.close()
    print("Finished stability")

    # 6. Console summary + text file
    summary_path = f"{outdir}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Feature Analysis Summary ===\n\n")
        f.write(f"Remaining features after correlation pruning: {len(X_corr.columns)}\n")
        f.write(f"Dropped correlated features: {len(dropped_corr)}\n\n")
        f.write("Top 10 Mutual Information:\n")
        f.write(str(mi.head(10)) + "\n\n")
        f.write("Top 10 RF Importance:\n")
        f.write(str(tree_imp.head(10)) + "\n\n")
        f.write("Top 10 Permutation Importance:\n")
        f.write(str(perm_imp.head(10)) + "\n\n")
        if stability is not None:
            f.write("Top 10 Stability Scores:\n")
            f.write(str(stability["stability_score"].head(10)) + "\n\n")

    print(f"Feature analysis complete. Results saved to: {outdir}")

    return {
        "remaining_features": list(X_corr.columns),
        "dropped_correlated": dropped_corr,
        "mutual_info": mi,
        "tree_importance": tree_imp,
        "permutation_importance": perm_imp,
        "shap_values": shap_values,
        "stability": stability,
        "corr_matrix": corr_matrix
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_series(series, path_csv, path_png=None, title=None):
    ensure_dir(os.path.dirname(path_csv))
    series.to_csv(path_csv)

    if path_png:
        plt.figure(figsize=(10, 6))
        series.head(40).plot(kind="bar")  # top 40 for readability
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path_png)
        plt.close()


def save_dataframe(df, path_csv):
    ensure_dir(os.path.dirname(path_csv))
    df.to_csv(path_csv)


def save_json(obj, path_json):
    ensure_dir(os.path.dirname(path_json))
    with open(path_json, "w") as f:
        json.dump(obj, f, indent=2)


def save_corr_heatmap(corr, path_png):
    ensure_dir(os.path.dirname(path_png))
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def save_shap_summary(shap_values, X, path_png):
    ensure_dir(os.path.dirname(path_png))
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()


def save_shap_bar(shap_values, X, path_png):
    ensure_dir(os.path.dirname(path_png))
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()


def build_ranked_feature_sets(results, outdir):
    mi = results["mutual_info"]
    rf_imp = results["tree_importance"]
    perm = results["permutation_importance"]
    stability = results["stability"]["stability_score"] if results["stability"] is not None else None

    # Normalize everything to comparable scales
    mi_rank_inv = (1 / (mi.rank() + 1))
    rf_norm = rf_imp / rf_imp.max()
    perm_norm = perm / perm.max()
    shap_mean = pd.Series(np.abs(results["shap_values"][1]).mean(axis=0), index=mi.index)
    shap_norm = shap_mean / shap_mean.max()
    stab_norm = stability / stability.max() if stability is not None else None

    # Core stable features
    core_score = 0.6 * stab_norm + 0.4 * mi_rank_inv if stability is not None else mi_rank_inv
    core_score = core_score.sort_values(ascending=False)
    core_score.to_csv(f"{outdir}/ranked/core_features.csv")

    # Tree-friendly features
    tree_score = (0.4 * rf_norm + 0.4 * shap_norm + 0.2 * mi_rank_inv).sort_values(ascending=False)
    tree_score.to_csv(f"{outdir}/ranked/tree_features.csv")

    # NN-friendly features
    if stability is not None:
        nn_score = (0.5 * mi_rank_inv + 0.3 * stab_norm + 0.2 * perm_norm).sort_values(ascending=False)
    else:
        nn_score = (0.7 * mi_rank_inv + 0.3 * perm_norm).sort_values(ascending=False)
    nn_score.to_csv(f"{outdir}/ranked/nn_features.csv")

    # Stacking-friendly features
    if stability is not None:
        stack_score = (0.5 * perm_norm + 0.3 * mi_rank_inv + 0.2 * stab_norm).sort_values(ascending=False)
    else:
        stack_score = (0.6 * perm_norm + 0.4 * mi_rank_inv).sort_values(ascending=False)
    stack_score.to_csv(f"{outdir}/ranked/stack_features.csv")

    # Console summary
    print("\n=== Ranked Feature Sets ===")
    print("Top 10 Core Stable Features:")
    print(core_score.head(10))
    print("\nTop 10 Tree-Friendly Features:")
    print(tree_score.head(10))
    print("\nTop 10 NN-Friendly Features:")
    print(nn_score.head(10))
    print("\nTop 10 Stacking-Friendly Features:")
    print(stack_score.head(10))

    return {
        "core": core_score,
        "tree": tree_score,
        "nn": nn_score,
        "stack": stack_score
    }


outdir = "C:\\Users\\gppal\\PycharmProjects\\marchmadness\\featureanalysis\\feature_analysis_output"
results = run_feature_analysis(
    path="C:\\Users\\gppal\\PycharmProjects\\marchmadness\\scraping\\data\\training_data.pckl",
    label_col="favorite_label",
    season_col="year",
    outdir=outdir
)
print(results)

# Write the ranked sets to a pickle file
with open(outdir+"\\results.pkl", "wb") as f:
    pickle.dump(results, f)

ranked_sets = build_ranked_feature_sets(results, outdir)

# Write the ranked sets to a pickle file
with open(outdir+"\\ranked_sets.pkl", "wb") as f:
    pickle.dump(ranked_sets, f)
