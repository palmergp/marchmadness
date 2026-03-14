"""Given a model, run a monte carlo to determine probabilities for each team to make different rounds. Then,
use those probabilities to create the highest value bracket"""
from bracket_predictor import BracketPredictor, EfficientBracketPredictor
from datetime import datetime


def main2(model, year, iterations):
    start_time = datetime.now()
    # Create a bracket predictor
    bp = BracketPredictor(model=model, year=year, probabilistic=True, silent=True)
    # bp = EfficientBracketPredictor(model=model, year=year)
    # Create a system for storing team results
    team_result = {}
    # Create a loop for running predictions
    print("Beginning monte carlo loop")
    for i in range(0, iterations):
        a, b, bracket = bp.main(False, False)
        # Add each team to team_result if its not there yet (yes this could be done better)
        if not team_result:
            for t in [x["team"] for x in bracket[1]]:
                team_result[t] = {
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0
                }
        # Increment the count for every team that made it to later rounds
        for r in range(2, 8):
            for team in [x["team"] for x in bracket[r]]:
                team_result[team][r] = team_result[team][r] + 1
    # Normalize to percentages
    print("Normalizing results")
    team_result_norm = {}
    for team in team_result:
        team_result_norm[team] = {}
        for r in range(2, 8):
            team_result_norm[team][r] = team_result[team][r]/iterations
    print(f"Finished {iter} iterations in {(datetime.now() - start_time).seconds} seconds")
    print("Done!")

from multiprocessing import Pool
from datetime import datetime

# Global predictor instance for each worker
_global_bp = None

def _init_worker(model, year):
    """Initializer that runs once per worker process."""
    global _global_bp
    _global_bp = BracketPredictor(model=model, year=year, probabilistic=True, silent=True)

def _run_single_simulation(_):
    """Runs one Monte Carlo simulation using the worker's predictor."""
    _, _, bracket = _global_bp.main(False, False)
    return bracket


def main(model, year, iterations, processes=8):
    start_time = datetime.now()

    print("Beginning parallel Monte Carlo loop")

    # --- Run simulations in parallel ---
    with Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(model, year)
    ) as pool:
        brackets = pool.map(_run_single_simulation, range(iterations))

    # --- Aggregate results ---
    team_result = {}

    # Initialize team_result using the first bracket
    first_bracket = brackets[0]
    for t in [x["team"] for x in first_bracket[1]]:
        team_result[t] = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

    # Count appearances across all brackets
    for bracket in brackets:
        for r in range(2, 8):
            for team in [x["team"] for x in bracket[r]]:
                team_result[team][r] += 1

    # --- Normalize ---
    print("Normalizing results")
    team_result_norm = {
        team: {r: team_result[team][r] / iterations for r in range(2, 8)}
        for team in team_result
    }

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Finished {iterations} iterations in {elapsed:.2f} seconds")
    print("Done!")

    return team_result_norm


if __name__ == '__main__':
    version = "v26_2_0_tree80"
    path = f"models/models26/{version}/"
    model_pkg = f"Random_Forest_{version}.package"
    year = 2025
    main2(model=path+model_pkg, year=year, iterations=10000)
