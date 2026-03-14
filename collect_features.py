import datetime
import json
import os
import pickle
import pandas as pd
from scraping.get_tournament_data import get_tournament_data
from scraping.get_team_data import get_team_stats
from scraping.get_schedule_data import get_schedule_stats
from scraping.get_roster_data import get_roster_stats
from scraping.get_kenpom_data import get_kenpom_stats

name_dict = {
    "UNC Asheville": "UNC-ASHEVILLE",
    "Texas A&M": "TEXAS-A&M",
    "UNC Wilmington": "UNC-WILMINGTON",
    "UNC Greensboro": "UNC-GREENSBORO",
    "Middle Tennessee State": "MIDDLE-TENNESSEE",
    "Miami (Fla.)": "MIAMI-FL",
    "Stephen F. Austin": "STEPHEN-F-AUSTIN",
    "UC Berkeley": "CALIFORNIA",
    "St. Mary's (Cal.)": "SAINT-MARYS-CA",
    "saint marys": "SAINT-MARYS-CA",
    "Saint Marys": "SAINT-MARYS-CA",
    "Saint Mary's": "SAINT-MARYS-CA",
    "St. Mary's": "SAINT-MARYS-CA",
    "Mount St. Mary's": "MOUNT-ST-MARYS",
    "Penn": "PENNSYLVANIA",
    "Long Island": "LONG-ISLAND-UNIVERSITY",
    "St. Bonaventure": "ST-BONAVENTURE",
    "Loyola (Ill.)": "ILLINOIS-CHICAGO",
    "Murray St.": "MURRAY-STATE",
    "St. John's": "ST-JOHNS-NY",
    'St. John\'s (NY)': "ST-JOHNS-NY",
    'st johns': "ST-JOHNS-NY",
    'St Johns': "ST-JOHNS-NY",
    "Saint Johns": "ST-JOHNS-NY",
    "LSU": "LOUISIANA-STATE",
    "lsu": "LOUISIANA-STATE",
    "UNLV": "NEVADA-LAS-VEGAS",
    "UTEP": "TEXAS-EL-PASO",
    "Detroit": "DETROIT-MERCY",
    "Loyola (Md.)": "LOYOLA-MD",
    "North Carolina A&T": "NORTH-CAROLINA-A&T",
    "Albany (N.Y.)": "ALBANY-NY",
    "USC": "SOUTHERN-CALIFORNIA",
    "usc": "SOUTHERN-CALIFORNIA",
    "ETSU": "EAST-TENNESSEE-STATE",
    "etsu": "EAST-TENNESSEE-STATE",
    "UNC": "NORTH-CAROLINA",
    "unc": "NORTH-CAROLINA",
    "LIU": "LONG-ISLAND-UNIVERSITY",
    "UCONN": "CONNECTICUT",
    "UConn": "CONNECTICUT",
    "VCU": "VIRGINIA-COMMONWEALTH",
    "vcu": "VIRGINIA-COMMONWEALTH",
    "Southern Miss": "SOUTHERN-MISSISSIPPI",
    "Loyola (MD)": "LOYOLA-MD",
    "BYU": "BRIGHAM-YOUNG",
    "byu": "BRIGHAM-YOUNG",
    "Miami (FL)": "MIAMI-FL",
    "Pitt": "PITTSBURGH",
    "Ole Miss": "MISSISSIPPI",
    "ole miss": "MISSISSIPPI",
    "UMass": "MASSACHUSETTS",
    "SMU": "SOUTHERN-METHODIST",
    "UMBC": "MARYLAND-BALTIMORE-COUNTY",
    "UCF": "CENTRAL-FLORIDA",
    "UCSB": "UC-SANTA-BARBARA",
    "FDU": "FAIRLEIGH-DICKINSON",
    "Albany": "ALBANY-NY",
    "Arkansas Little Rock": "LITTLE-ROCK",
    "Bowling Green": "BOWLING-GREEN-STATE",
    "Centenary": "CENTENARY-LA",
    "Central Connecticut": "CENTRAL-CONNECTICUT-STATE",
    "FIU": "FLORIDA-INTERNATIONAL",
    "Grambling St.": "GRAMBLING",
    "Houston Baptist": "HOUSTON-CHRISTIAN",
    "IPFW": "PURDUE-FORT-WAYNE",
    "LIU Brooklyn": "LONG-ISLAND-UNIVERSITY",
    "Louisiana Lafayette": "LOUISIANA",
    "Loyola Chicago": "LOYOLA-IL",
    "Prairie View A&M": "PRAIRIE-VIEW",
    "St. Francis NY": "ST-FRANCIS-NY",
    "Sam Houston St.": "SAM-HOUSTON",
    "Texas A&M Corpus Chris": "TEXAS-A&M-CORPUS-CHRISTI",
    "Texas Pan American": "TEXAS-RIO-GRANDE-VALLEY",
    "UMKC": "KANSAS-CITY",
    "USC Upstate": "SOUTH-CAROLINA-UPSTATE",
    "Charleston": "COLLEGE-OF-CHARLESTON",
    "Nebraska Omaha": "OMAHA",
    "Cal Baptist": "CALIFORNIA-BAPTIST",
    "UMass Lowell": "MASSACHUSETTS-LOWELL",
    "Queens": "QUEENS-NC",
    "St. Thomas": "ST-THOMAS",
    "UT Rio Grande Valley": "TEXAS-RIO-GRANDE-VALLEY",
    "sdsu": "SAN-DIEGO-STATE",
    "SDSU": "SAN-DIEGO-STATE",
    "mcneese": "MCNEESE-STATE",
    "Mcneese": "MCNEESE-STATE",
    "McNeese": "MCNEESE-STATE",
    "Texas San Antonio": "UTSA",
    "NC Asheville": "UNC-ASHEVILLE",
    "North Carolina St.": "NC-STATE",
    "Texas A&M; Corpus Chris": "TEXAS-A&M-CORPUS-CHRISTI",
    "Wisconsin Milwaukee": "MILWAUKEE",
    "Wisconsin Green Bay": "GREEN-BAY",
    "Texas Christian": "TCU",
    "IUPU Fort Wayne": "PURDUE-FORT-WAYNE",
    "Missouri Kansas City": "KANSAS-CITY",
    "MD Baltimore County": "MARYLAND-BALTIMORE-COUNTY",
    "NJ Inst Of Technology": "NJIT",
    "Prarie View A&M;": "PRAIRIE-VIEW",
    "Prarie View": "PRAIRIE-VIEW",
    "NC Wilmington": "UNC-WILMINGTON",
    "NC Greensboro": "UNC-GREENSBORO",
    "Texas Arlington": "UT-ARLINGTON",
    "Virginia Military Inst": "VMI",
    "SIUE": "SIU-EDWARDSVILLE",
    "siue": "SIU-EDWARDSVILLE",
    "mount saint marys": "MOUNT-ST-MARYS"
}


def reformat_name(name):
    """Fixes names with multiple spellings"""
    try:
        name = name_dict[name]
    except KeyError:
        name = name.replace(" ", "-")
        if name.startswith("St."):
            name = name.replace("St.", "SAINT")
        name = name.replace("St.", "STATE")
        name = name.replace("St.", "SAINT")
        name = name.replace("'", "")
        name = name.replace(".", "")
        #name = name.replace("UC-", "CALIFORNIA-")
        name = name.replace("(", "")
        name = name.replace(")", "")
        name = name.replace(";", "")
    return name.upper()


def calculate_combo_features(game_row):
    """Given a game row full of both team's stats, return the custom calculated statistics"""
    combo_features = {}
    # Add the difference features
    for col in game_row.columns:
        if col.startswith("favorite_") and not isinstance(game_row[col][0], str):
            base = col.replace("favorite_", "")
            dog_col = f"underdog_{base}"

            if dog_col in game_row.columns:
                diff_name = f"{base}_diff"
                combo_features[diff_name] = game_row[dog_col] - game_row[col]
    # Add matchup synergy features
    # Offensive rebounding vs defensive rebounding
    combo_features["underdog_ORB_synergy"] = (
            game_row["underdog_offensive_rebound_percentage"] -
            game_row["favorite_opp_offensive_rebound_percentage"]
    )

    # Turnover pressure vs ball security
    combo_features["underdog_tov_synergy"] = (
            game_row["underdog_opp_turnover_percentage"] -
            game_row["favorite_turnover_percentage"]
    )

    # Three-point shooting vs three-point defense
    combo_features["underdog_3pt_synergy"] = (
            game_row["underdog_three_point_field_goal_percentage"] -
            game_row["favorite_opp_three_point_field_goal_percentage"]
    )

    # Pace advantage (positive = underdog speeds up the game)
    combo_features["underdog_tempo_advantage"] = (
            game_row["underdog_pace"] - game_row["favorite_pace"]
    )

    # Efficiency mismatch
    combo_features["underdog_efficiency_synergy"] = (
            game_row["underdog_ORtg"] - game_row["favorite_DRtg"]
    )

    # Variance (3PA rate)
    combo_features["underdog_variance_synergy"] = (
            game_row["underdog_three_point_attempt_rate"] -
            game_row["favorite_three_point_attempt_rate"]
    )

    # Schedule toughness mismatch
    combo_features["underdog_recent_efficiency_synergy"] = (
            game_row["underdog_last_10_win_percentage"] * game_row["underdog_NetRtg"]
            - game_row["favorite_last_10_win_percentage"] * game_row["favorite_NetRtg"]
    )
    combo_features["underdog_physicality_synergy"] = (
                                                             game_row["underdog_weighted_avg_weight"] -
                                                             game_row["favorite_weighted_avg_weight"]
                                                     ) + (
                                                             game_row["underdog_offensive_rebound_percentage"] -
                                                             game_row["favorite_opp_offensive_rebound_percentage"]
                                                     )
    combo_features["underdog_chaos_factor"] = (
            game_row["underdog_opp_turnover_percentage"] +
            game_row["underdog_three_point_attempt_rate"] -
            game_row["favorite_turnover_percentage"]
    )
    combo_features["favorite_control_factor"] = (
            game_row["favorite_pace"] -
            game_row["underdog_pace"] +
            game_row["favorite_turnover_percentage"]
    )
    # Add four factor features
    combo_features["favorite_four_factor_score"] = (
            0.4 * game_row["favorite_effective_field_goal_percentage"] +
            0.25 * game_row["favorite_turnover_percentage"] +
            0.2 * game_row["favorite_offensive_rebound_percentage"] +
            0.15 * game_row["favorite_free_throws_per_field_goal_attempt"]
    )

    combo_features["underdog_four_factor_score"] = (
            0.4 * game_row["underdog_effective_field_goal_percentage"] +
            0.25 * game_row["underdog_turnover_percentage"] +
            0.2 * game_row["underdog_offensive_rebound_percentage"] +
            0.15 * game_row["underdog_free_throws_per_field_goal_attempt"]
    )
    combo_features["favorite_def_four_factor_score"] = (
            0.4 * game_row["favorite_opp_effective_field_goal_percentage"] +
            0.25 * game_row["favorite_opp_turnover_percentage"] +
            0.2 * game_row["favorite_opp_offensive_rebound_percentage"] +
            0.15 * game_row["favorite_opp_free_throws_per_field_goal_attempt"]
    )

    combo_features["underdog_def_four_factor_score"] = (
            0.4 * game_row["underdog_opp_effective_field_goal_percentage"] +
            0.25 * game_row["underdog_opp_turnover_percentage"] +
            0.2 * game_row["underdog_opp_offensive_rebound_percentage"] +
            0.15 * game_row["underdog_opp_free_throws_per_field_goal_attempt"]
    )
    combo_features["favorite_3pt_volatility"] = (
            game_row["favorite_three_point_attempt_rate"] *
            (1 - game_row["favorite_three_point_field_goal_percentage"])
    )

    combo_features["underdog_3pt_volatility"] = (
            game_row["underdog_three_point_attempt_rate"] *
            (1 - game_row["underdog_three_point_field_goal_percentage"])
    )
    combo_features["favorite_paint_dominance"] = (
            game_row["favorite_two_point_field_goal_percentage"] -
            game_row["favorite_opp_two_point_field_goal_percentage"] +
            game_row["favorite_offensive_rebound_percentage"]
    )

    combo_features["underdog_paint_dominance"] = (
            game_row["underdog_two_point_field_goal_percentage"] -
            game_row["underdog_opp_two_point_field_goal_percentage"] +
            game_row["underdog_offensive_rebound_percentage"]
    )
    return combo_features


def collect_features(recalculate=False):
    """This function compiles the features for all games to be used as training data
    Input:
        - recalculate: (bool) flag to indicate if features should be recalculated or not. If set to false, the function
                        will only calculate features if they are not already saved on the machine
    Output:
        - training_data: (DataFrame) Dataframe containing all training data across all years
    """

    # Load previous training data if it exists
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "scraping")
    full_path = os.path.join(full_path, "data")
    filename = os.path.join(full_path, f"training_data.pckl")
    feature_filename = os.path.join(full_path, f"full_featurenames.json")
    # See if the training data already exists
    if os.path.isfile(filename) and not recalculate:
        with open(filename, "rb") as f:
            training_data = pickle.load(f)
    else:
        # If the file doesn't exist or the recalculate-flag was raised,
        # start collecting training data
        print("Recalculating training data")
        training_data = pd.DataFrame()

        # Get the current year
        curr_year = datetime.datetime.now().year
        # If it is past april, then the tournament is over then we can include this year too
        if datetime.datetime.now().month > 4:
            curr_year = curr_year + 1
        # Get data from 2011 to last year
        for year in range(2011, curr_year):

            # Due to COVID, the 2020 tournament was canceled, therefore, it cannot be used for training
            if year == 2020:
                continue

            # Check if the year's data is already in training data
            if not training_data.empty and training_data['year'].isin([year]).any():
                continue

            # Get all team stats from that year
            all_team_stats = get_team_stats(year)

            # Fix the names
            all_team_stats.index = all_team_stats.index.map(reformat_name)

            # Get all kenpom stats
            all_kenpom_stats = get_kenpom_stats(year)
            all_kenpom_stats.index = all_kenpom_stats.index.map(reformat_name)


            # If the year's data is not in the set, go collect it
            # First get all tournament matchups from the year
            bracket_data = get_tournament_data(year)
    
            # Loop through each game
            for index, game in bracket_data.iterrows():
                # Process the favorite first
                if int(game["winning_team_seed"]) < int(game["losing_team_seed"]):
                    favorite = {"name": game["winning_team"],
                                "seed": game["winning_team_seed"],
                                "type": "favorite",
                                "label": "expected"}
                    underdog = {"name": game["losing_team"],
                                "seed": game["losing_team_seed"],
                                "type": "underdog",
                                "label": "expected"}
                else:
                    favorite = {"name": game["losing_team"],
                                "seed": game["losing_team_seed"],
                                "type": "favorite",
                                "label": "upset"}
                    underdog = {"name": game["winning_team"],
                                "seed": game["winning_team_seed"],
                                "type": "underdog",
                                "label": "upset"}
                game_row = pd.DataFrame()
                for team in [favorite, underdog]:
                    # Get all of the features for that team
                    team_stats = all_team_stats.loc[[reformat_name(team["name"])]]
                    schedule_stats = get_schedule_stats([reformat_name(team["name"])], year).loc[[reformat_name(team["name"])]]
                    roster_stats = get_roster_stats([reformat_name(team["name"])], year).loc[[reformat_name(team["name"])]]
                    tourney_stats = pd.DataFrame({"seed": [int(team["seed"])],
                                                  "label": favorite["label"]})
                    kenpom_stats = all_kenpom_stats.loc[[reformat_name(team["name"])]]

                    # Combine all of these stats into a single dataframe row
                    team_row = pd.concat([tourney_stats,
                                          team_stats.rename(index={reformat_name(team["name"]):0}),
                                          schedule_stats.rename(index={reformat_name(team["name"]):0}),
                                          roster_stats.rename(index={reformat_name(team["name"]):0}),
                                          kenpom_stats.rename(index={reformat_name(team["name"]):0})], axis=1)
                    # Update the column names to say favorite or underdog
                    team_row = team_row.add_prefix(team["type"] + "_")
                    # Append to the game row
                    game_row = pd.concat([game_row, team_row], axis=1)
                # Before proceeding, use kenpom as tiebreaker when seeds are the same
                if int(game["winning_team_seed"]) == int(game["losing_team_seed"]) and game_row["underdog_kenpomRank"][0] < game_row["favorite_kenpomRank"][0]:
                    rename_map = {}
                    for col in game_row.columns:
                        if col.startswith("favorite_"):
                            new_name = col.replace("favorite_", "underdog_", 1)
                            rename_map[col] = new_name
                        elif col.startswith("underdog_"):
                            new_name = col.replace("underdog_", "favorite_", 1)
                            rename_map[col] = new_name
                    game_row = game_row.rename(columns=rename_map)
                    if game_row["favorite_label"][0] == "upset":
                        game_row["favorite_label"][0] = "expected"
                        game_row["underdog_label"][0] = "expected"
                    elif game_row["favorite_label"][0] == "expected":
                        game_row["favorite_label"][0] = "upset"
                        game_row["underdog_label"][0] = "upset"
                # Calculate diff, synergy, and composite features
                combo_features = calculate_combo_features(game_row)
                # Add into dataframe
                game_row = pd.concat([game_row, pd.DataFrame([combo_features])], axis=1)

                # Add the year
                game_row = game_row.assign(year=year)
                # Add the round
                game_row = game_row.assign(round=game["round"])
                # Append the game row to the overall data
                training_data = pd.concat([training_data, game_row])
        # Save training data
        with open(filename, "wb") as f:
            pickle.dump(training_data, f)
        # Save Full Feature list
        all_cols = training_data.columns.to_list()
        feature_names = {
            "individual_stats": [],
            "difference_stats": [],
            "synergy_stats": [],
            "other": []
        }
        all_cols = [x for x in all_cols if x not in ["year", "favorite_label", "underdog_label"]]
        for col in all_cols:
            if col in combo_features:
                if "diff" in col:
                    feature_names["difference_stats"].append(col)
                else:
                    feature_names["synergy_stats"].append(col)
            else:
                if "favorite" in col or "underdog" in col:
                    basic_name = col.replace("favorite_","").replace("underdog_","")
                    if basic_name not in feature_names["individual_stats"]:
                        feature_names["individual_stats"].append(basic_name)
                else:
                    feature_names["other"].append(col)

        with open(feature_filename, "w") as f:
            json.dump(feature_names, f, indent=2)
    return training_data
            

if __name__ == "__main__":
    t_data = collect_features(True)


