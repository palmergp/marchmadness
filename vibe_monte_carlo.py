import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

from scraping.get_tournament_data import get_tournament_data
from matchup_predictor import MatchupPredictor

from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------
year = 2026
version = "v26_5_0_tree_early_top80"
path = f"models/models26/{version}/"
model_pkg = f"Random_Forest_{version}.package"
model_path = path + model_pkg
late_version = "v26_5_0_tree_late_top80"
late_model_pkg = f"Random_Forest_{late_version}.package"
late_model_path = f"models/models26/{late_version}/" + late_model_pkg

# -----------------------------
# Load and reorder bracket
# -----------------------------
df = get_tournament_data(year)

# Handle 2021 forfeit
if year == 2021:
    new_row = pd.DataFrame(
        {
            'winning_team': "Oregon",
            'winning_team_seed': 7,
            'losing_team': "VCU",
            "losing_team_seed": 10,
            "round": 1,
            "winning_team_score": 0,
            "losing_team_score": 0
        },
        index=[51]
    )
    df = pd.concat([df.iloc[:51], new_row, df.iloc[51:]]).reset_index(drop=True)

# Reorder rows into bracket order
if len(df) == 32:  # only round 1 available
    new_order = list(range(16, 32)) + list(range(0, 16))
else:
    new_order = list(range(0, 15)) + list(range(45, 60)) + list(range(15, 45)) + list(range(60, 63))

if year != 2025:
    df = df.reindex(new_order).reset_index(drop=True)


# -----------------------------
# Deterministic team ID mapping
# -----------------------------
teams = sorted(set(df["winning_team"]).union(df["losing_team"]))
team_to_id = {team: i for i, team in enumerate(teams)}
id_to_team = {i: team for team, i in team_to_id.items()}

# Extract seeds from round 1
team_seed = {}
for _, row in df.iterrows():
    if row["round"] == 1:
        team_seed[row["winning_team"]] = row["winning_team_seed"]
        team_seed[row["losing_team"]] = row["losing_team_seed"]


# -----------------------------
# Build bracket template
# -----------------------------
def build_bracket_template(df, team_to_id):
    template = defaultdict(list)
    for _, row in df.iterrows():
        r = int(row["round"])
        t1 = team_to_id[row["winning_team"]]
        t2 = team_to_id[row["losing_team"]]
        template[r].append((t1, t2))
    return template


bracket_template = build_bracket_template(df, team_to_id)


# -----------------------------
# Precompute probability matrix
# -----------------------------
if late_model_path:
    predictor = MatchupPredictor(model=model_path, late_model=late_model_path, round_split=1, silent=True)
else:
    predictor = MatchupPredictor(model=model_path, silent=True)
predictor.set_year(year)

# Try to open a saved version of the predictions
if late_model_path:
    saved_predictions_path = model_path.replace(".package", "_") + str(year) + "_" + late_model_pkg.replace(".package", "_") + "predictions.pkl"
else:
    saved_predictions_path = model_path.replace(".package", "_")+str(year)+"predictions.pkl"

try:
    with open(saved_predictions_path, "rb") as f:
        P = pickle.load(f)
except FileNotFoundError:
    N = len(team_to_id)
    P = np.zeros((N, N), dtype=float)

    count = 0
    predict_start = datetime.now()
    total_time = 0
    for team1 in teams:
        time_remaining = ((datetime.now() - predict_start).seconds / count) * (len(teams) - count) if count > 0 else "unknown"
        count += 1
        print(f"Starting {count} of {len(teams)}. {time_remaining} seconds remaining")
        for team2 in teams:
            if team1 == team2:
                continue

            t1 = team_to_id[team1]
            t2 = team_to_id[team2]

            s1 = team_seed[team1]
            s2 = team_seed[team2]

            _, p = predictor.predict(team1, s1, team2, s2, 1, year)
            P[t1, t2] = p[0]
            P[t2, t1] = p[1]

    with open(saved_predictions_path, "wb") as f:
        pickle.dump(P, f)

# -----------------------------
# Single simulation
# -----------------------------
def simulate_bracket(P, template, rng):
    winners = {1: []}

    # Round 1
    for t1, t2 in template[1]:
        p = P[t1, t2]
        rand_num = rng.random()
        winners[1].append(t1 if rand_num < p else t2)

    # Rounds 2–6
    for r in range(2, 7):
        prev = winners[r - 1]
        winners[r] = []
        for i in range(0, len(prev), 2):
            t1 = prev[i]
            t2 = prev[i + 1]
            p = P[t1, t2]
            winners[r].append(t1 if rng.random() < p else t2)

    return winners


# -----------------------------
# Parallel Monte Carlo
# -----------------------------
_global_P = None
_global_template = None

def init_worker(P, template):
    global _global_P, _global_template
    _global_P = P
    _global_template = template

def worker_run(n):
    rng = np.random.default_rng()
    # Pre-initialize counts for all teams
    counts = {t: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} for t in range(len(id_to_team))}

    for _ in range(n):
        winners = simulate_bracket(_global_P, _global_template, rng)
        for r in range(1, 7):
            for t in winners[r]:
                counts[t][r] += 1

    return counts


def merge_counts(list_of_counts):
    final = {}
    for counts in list_of_counts:
        for team, rounds in counts.items():
            if team not in final:
                final[team] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            for r in rounds:
                final[team][r] += rounds[r]
    return final

def run_monte_carlo(P, template, iterations, processes=8):
    sims_per_worker = iterations // processes
    extra = iterations % processes
    jobs = [sims_per_worker + (1 if i < extra else 0) for i in range(processes)]

    with Pool(processes=processes, initializer=init_worker, initargs=(P, template)) as pool:
        results = pool.map(worker_run, jobs)

    final_counts = merge_counts(results)

    normalized = {
        team: {r: final_counts[team][r] / iterations for r in range(1, 7)}
        for team in final_counts
    }

    return normalized


def seed_prior(seed_fav, seed_dog):
    # historical win probability by seed difference
    diff = seed_dog - seed_fav
    # smaller diff → closer game → closer to 0.5
    # larger diff → favorite more likely
    return 1 / (1 + 10 ** (diff / 4))


def build_optimal_bracket(template, id_results, points, team_seeds, mode="total"):
    """

    :param template:
    :param id_results:
    :param points:
    :param mode: indicates if EV should be calculated for total EV or round EV. Should be a string of either
                "total" or "round"
    :return:
    """
    optimal = {}
    round_lambda = {
        1: 0.95,
        2: 0.9,
        3: 0.8,
        4: 0.7,
        5: 0.6,
        6: 0.5
    }

    # Round 1
    optimal[1] = []
    for t1, t2 in template[1]:
        p1_model = id_results[t1][2]
        p2_model = id_results[t2][2]

        # get seeds
        s1 = team_seeds[id_to_team[t1]]
        s2 = team_seeds[id_to_team[t2]]

        # compute priors
        p1_prior = seed_prior(min(s1,s2), max(s1,s2))
        p2_prior = 1 - p1_prior

        # blend by round
        lam = round_lambda[1]  # e.g., {1:0.95, 2:0.90, ..., 6:0.50}

        p1 = lam * p1_model + (1 - lam) * p1_prior
        p2 = lam * p2_model + (1 - lam) * p2_prior

        if mode == "total":
            ev1 = sum(points[1] * id_results[t1][k + 1] for k in range(1, 6))
            ev2 = sum(points[1] * id_results[t2][k + 1] for k in range(1, 6))
        else:
            ev1 = points[1] * p1
            ev2 = points[1] * p2
        optimal[1].append(t1 if ev1 >= ev2 else t2)

    # Rounds 2–6
    for r in range(2, 7):
        optimal[r] = []
        prev = optimal[r - 1]
        for i in range(0, len(prev), 2):
            t1 = prev[i]
            t2 = prev[i + 1]

            # Probability of reaching the NEXT round
            next_round = r + 1 if r < 6 else r
            p1_model = id_results[t1][next_round]
            p2_model = id_results[t2][next_round]

            # get seeds
            s1 = team_seeds[id_to_team[t1]]
            s2 = team_seeds[id_to_team[t2]]

            # compute priors
            p1_prior = seed_prior(min(s1, s2), max(s1, s2))
            p2_prior = 1 - p1_prior

            # blend by round
            lam = round_lambda[r]  # e.g., {1:0.95, 2:0.90, ..., 6:0.50}

            p1 = lam * p1_model + (1 - lam) * p1_prior
            p2 = lam * p2_model + (1 - lam) * p2_prior

            if mode == "total":
                ev1 = sum(points[k] * id_results[t1][k + 1] for k in range(r, 6))
                ev2 = sum(points[k] * id_results[t2][k + 1] for k in range(r, 6))
            else:
                ev1 = points[r] * p1
                ev2 = points[r] * p2

            optimal[r].append(t1 if ev1 >= ev2 else t2)

    return optimal

def convert_optimal_to_names(optimal, id_to_team):
    named = {}
    for r, teams in optimal.items():
        named[r] = [id_to_team[t] for t in teams]
    return named

def optimal_bracket_to_df(optimal_ids, id_to_team):
    rows = []

    for r in range(1, 7):
        teams = optimal_ids[r]

        # Pair teams into matchups
        for i in range(0, len(teams), 2):
            t1 = teams[i]
            t2 = teams[i+1]

            # Winner is whichever team advanced in the NEXT round
            if r < 6:
                next_round_teams = optimal_ids[r+1]
                winner = t1 if t1 in next_round_teams else t2
            else:
                # Championship round: only one winner
                winner = teams[0]

            rows.append({
                "round": r,
                "team1": id_to_team[t1],
                "team2": id_to_team[t2],
                "winner": id_to_team[winner]
            })

    return pd.DataFrame(rows)

def score_bracket(optimal_ids, df, team_to_id, points):
    score = 0
    scores_by_round = []

    # Group actual results by round in the same order as the template
    actual_by_round = defaultdict(list)
    for _, row in df.iterrows():
        r = int(row["round"])
        winner_id = team_to_id[row["winning_team"]]
        actual_by_round[r].append(winner_id)

    # Compare predicted winners to actual winners
    for r in range(1, 7):
        round_score = 0
        predicted_winners = optimal_ids[r]
        actual_winners = actual_by_round[r]

        # They should be the same length (same number of games)
        for pred in predicted_winners:
            if pred in actual_winners:
                score += points[r]
                round_score += points[r]
        scores_by_round.append(round_score)

    return score, scores_by_round


if __name__ == "__main__":
    start_time = datetime.now()
    # -----------------------------
    # Run Monte Carlo
    # -----------------------------
    print(f"Set up took {(datetime.now() - start_time).seconds}s")
    probs = run_monte_carlo(P, bracket_template, iterations=500000)

    # Map back to team names
    results = {id_to_team[t]: rounds for t, rounds in probs.items()}
    print(f"Total monte carlo took {(datetime.now() - start_time).seconds}s")

    # Convert results to ID-keyed
    id_results = {tid: results[id_to_team[tid]] for tid in id_to_team}

    # Define scoring system
    points = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

    # Build optimal bracket
    optimal_ids_total = build_optimal_bracket(bracket_template, id_results, points, team_seed, "total")
    optimal_ids_round = build_optimal_bracket(bracket_template, id_results, points, team_seed, "round")

    # Convert to names
    optimal_bracket_total = convert_optimal_to_names(optimal_ids_total, id_to_team)
    optimal_bracket_round = convert_optimal_to_names(optimal_ids_round, id_to_team)

    print(optimal_bracket_total)

    bracket_score_total, score_by_round_total = score_bracket(
        optimal_ids=optimal_ids_total,
        df=df,
        team_to_id=team_to_id,
        points=points
    )
    print("Bracket Score Total:", bracket_score_total)
    bracket_score_round, score_by_round_round = score_bracket(
        optimal_ids=optimal_ids_total,
        df=df,
        team_to_id=team_to_id,
        points=points
    )
    print("Bracket Score Round:", bracket_score_round)

    print("Done")
