import datetime
import os
import pickle
import pandas as pd
from get_tournament_data import get_tournament_data
from get_team_data import get_team_stats
from get_schedule_data import get_schedule_stats
from get_roster_data import get_roster_stats

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
    "Mount St. Mary's": "MOUNT-ST-MARYS",
    "Penn": "PENNSYLVANIA",
    "Long Island": "LONG-ISLAND-UNIVERSITY",
    "St. Bonaventure": "ST-BONAVENTURE",
    "Loyola (Ill.)": "ILLINOIS-CHICAGO",
    "Murray St.": "MURRAY-STATE",
    "St. John's": "ST-JOHNS-NY",
    'St. John\'s (NY)': "ST-JOHNS-NY",
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
    "LIU": "LONG-ISLAND-UNIVERSITY",
    "UCONN": "CONNECTICUT",
    "UConn": "CONNECTICUT",
    "VCU": "VIRGINIA-COMMONWEALTH",
    "Southern Miss": "SOUTHERN-MISSISSIPPI",
    "Loyola (MD)": "LOYOLA-MD",
    "BYU": "BRIGHAM-YOUNG",
    "Miami (FL)": "MIAMI-FL",
    "Pitt": "PITTSBURGH",
    "Ole Miss": "MISSISSIPPI",
    "UMass": "MASSACHUSETTS",
    "SMU": "SOUTHERN-METHODIST",
    "UMBC": "MARYLAND-BALTIMORE-COUNTY",
    "UCF": "CENTRAL-FLORIDA",
    "UCSB": "UC-SANTA-BARBARA"
}


def reformat_name(name):
    """Fixes names with multiple spellings"""
    try:
        name = name_dict[name]
    except KeyError:
        name = name.replace(" ", "-")
        name = name.replace("St.", "SAINT")
        name = name.replace("'", "")
        #name = name.replace("UC-", "CALIFORNIA-")
        name = name.replace("(", "")
        name = name.replace(")", "")
    return name.upper()

def collect_features(feature_list):
    """This function compiles the features for all games to be used as training data"""

    # Load previous training data if it exists
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "data")
    filename = os.path.join(full_path, f"training_data.pckl")
    # First load the schedule data
    try:
        with open(filename, "rb") as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        print("Could not find file. Assuming this is the first attempt at getting schedule data")
        training_data = pd.DataFrame()

        # Get the current year
        curr_year = datetime.datetime.now().year
        # Get data from 2011 to last year
        for year in range(2011, curr_year):

            if year == 2020:
                continue
            # Check if the year's data is already in training data
            if not training_data.empty and training_data['year'].isin([year]).any():
                continue

            # Get all team stats from that year
            all_team_stats = get_team_stats(year)

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
                    # Combine all of these stats into a single dataframe row
                    team_row = pd.concat([tourney_stats,
                                          team_stats.rename(index={reformat_name(team["name"]):0}),
                                          schedule_stats.rename(index={reformat_name(team["name"]):0}),
                                          roster_stats.rename(index={reformat_name(team["name"]):0})], axis=1)
                    # Update the column names to say favorite or underdog
                    team_row = team_row.add_prefix(team["type"] + "_")
                    # Append to the game row
                    game_row = pd.concat([game_row, team_row], axis=1)
                # Add the seed diff feature and the year
                game_row = game_row.assign(seed_diff=game_row['underdog_seed'] - game_row['favorite_seed'])
                game_row = game_row.assign(year=year)
                # Append the game row to the overall data
                training_data = training_data.append(game_row)
        # Save training data
        with open(filename, "wb") as f:
            pickle.dump(training_data,f)
    return training_data
            

if __name__ == "__main__":
    featurenames = ['favorite_seed', 'underdog_seed', 'seed_diff', 'favorite_assist_percentage', 'underdog_assist_percentage', 'favorite_block_percentage', 'underdog_block_percentage', 'favorite_effective_field_goal_percentage', 'underdog_effective_field_goal_percentage', 'favorite_free_throw_percentage', 'underdog_free_throw_percentage', 'favorite_free_throws_per_field_goal_attempt', 'underdog_free_throws_per_field_goal_attempt', 'favorite_offensive_rating', 'underdog_offensive_rating', 'favorite_offensive_rebound_percentage', 'underdog_offensive_rebound_percentage', 'favorite_opp_effective_field_goal_percentage', 'underdog_opp_effective_field_goal_percentage', 'favorite_opp_free_throws_per_field_goal_attempt', 'underdog_opp_free_throws_per_field_goal_attempt', 'favorite_opp_offensive_rebound_percentage', 'underdog_opp_offensive_rebound_percentage', 'favorite_opp_three_point_field_goal_percentage', 'underdog_opp_three_point_field_goal_percentage', 'favorite_opp_two_point_field_goal_percentage', 'underdog_opp_two_point_field_goal_percentage', 'favorite_opp_total_rebound_percentage', 'underdog_opp_total_rebound_percentage', 'favorite_opp_true_shooting_percentage', 'underdog_opp_true_shooting_percentage', 'favorite_opp_turnover_percentage', 'underdog_opp_turnover_percentage', 'favorite_pace', 'underdog_pace', 'favorite_points', 'underdog_points', 'favorite_simple_rating_system', 'underdog_simple_rating_system', 'favorite_steal_percentage', 'underdog_steal_percentage', 'favorite_strength_of_schedule', 'underdog_strength_of_schedule', 'favorite_three_point_attempt_rate', 'underdog_three_point_attempt_rate', 'favorite_three_point_field_goal_percentage', 'underdog_three_point_field_goal_percentage', 'favorite_two_point_field_goal_percentage', 'underdog_two_point_field_goal_percentage', 'favorite_total_rebound_percentage', 'underdog_total_rebound_percentage', 'favorite_true_shooting_percentage', 'underdog_true_shooting_percentage', 'favorite_turnover_percentage', 'underdog_turnover_percentage', 'favorite_win_percentage', 'underdog_win_percentage', 'favorite_top5_total_per', 'underdog_top5_total_per', 'favorite_top_per_percentage', 'underdog_top_per_percentage', 'favorite_ranked_wins', 'underdog_ranked_wins', 'favorite_ranked_losses', 'underdog_ranked_losses', 'favorite_ranked_win_percentage', 'underdog_ranked_win_percentage', 'favorite_points_per_ranked', 'underdog_points_per_ranked', 'favorite_opp_points_per_ranked', 'underdog_opp_points_per_ranked', 'favorite_margin_of_vict_ranked', 'underdog_margin_of_vict_ranked']
    training_data = collect_features(featurenames)


