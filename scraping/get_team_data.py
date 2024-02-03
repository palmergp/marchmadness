import pickle
import pandas as pd
import time
from scraping.smart_request import smart_request
import os

NAME_LOOKUP = {
    "G": "games",
    "W": "wins",
    "L": "losses",
    "W-L%": "win_percentage",
    "SRS": "simple_rating_system",
    "SOS": "strength_of_schedule",
    "MP": "minutes_played",
    "FG": "field_goals",
    "FGA": "field_goal_attempts",
    "FG%": "field_goal_percentage",
    "3P":  "three_point_field_goals",
    "3PA": "three_point_field_goal_attempts",
    "3P%": "three_point_field_goal_percentage",
    "FT": "free_throws",
    "FTA": "free_throw_attempts",
    "FT%": "free_throw_percentage",
    "ORB": "offensive_rebounds",
    "TRB": "total_rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "PF": "personal_fouls",
    "Pace": "pace",
    "ORtg": "offensive_rating",
    "FTr": "free_throw_attempt_rate",
    "3PAr": "three_point_attempt_rate",
    "TS%": "true_shooting_percentage",
    "TRB%": "total_rebound_percentage",
    "AST%": "assist_percentage",
    "STL%": "steal_percentage",
    "BLK%": "block_percentage",
    "eFG%": "effective_field_goal_percentage",
    "TOV%": "turnover_percentage",
    "ORB%": "offensive_rebound_percentage",
    "FT/FGA": "free_throws_per_field_goal_attempt"
}

def has_number(input_string):
    for c in input_string:
        if not c.isdigit() and c != "." and c != "-":
            return False
    return True

def get_team_stats(year):
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "data")
    filename = os.path.join(full_path, f"roster_data_{year}.pckl")

    try:
        print("Loading Data")
        # Load from file
        filename = os.path.join(full_path, f"basic_stats_{year}.pckl")
        f = open(filename, "rb")
        school_stats = pickle.load(f)
        f.close()
        filename = os.path.join(full_path, f"advanced_stats_{year}.pckl")
        f = open(filename, "rb")
        adv_school_stats = pickle.load(f)
        f.close()
        filename = os.path.join(full_path, f"basic_opp_stats_{year}.pckl")
        f = open(filename, "rb")
        opp_school_stats = pickle.load(f)
        f.close()
        filename = os.path.join(full_path, f"advanced_opp_stats_{year}.pckl")
        f = open(filename, "rb")
        opp_adv_school_stats = pickle.load(f)
        f.close()
    except FileNotFoundError:
        print("Making requests for team data")
        # Get basic stats
        response = smart_request(f"https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html")
        school_stats = str(response.content)

        # Save as a pickle file to prevent getting blocked
        filename = os.path.join(full_path, f"basic_stats_{year}.pckl")
        f = open(filename, "wb")
        pickle.dump(school_stats,f)
        f.close()
        time.sleep(1)

        # Get advanced stats
        response = smart_request(f"https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html")
        adv_school_stats = str(response.content)

        # Save as a pickle file to prevent getting blocked
        filename = os.path.join(full_path, f"advanced_stats_{year}.pckl")
        f = open(filename, "wb")
        pickle.dump(adv_school_stats, f)
        f.close()
        time.sleep(1)

        # Get basic opp stats
        response = smart_request(f"https://www.sports-reference.com/cbb/seasons/men/{year}-opponent-stats.html")
        opp_school_stats = str(response.content)

        # Save as a pickle file to prevent getting blocked
        filename = os.path.join(full_path, f"basic_opp_stats_{year}.pckl")
        f = open(filename, "wb")
        pickle.dump(opp_school_stats, f)
        f.close()
        time.sleep(1)

        # Get advanced opponent stats
        response = smart_request(f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-opponent-stats.html")
        opp_adv_school_stats = str(response.content)

        # Save as a pickle file to prevent getting blocked
        filename = os.path.join(full_path, f"advanced_opp_stats_{year}.pckl")
        f = open(filename, "wb")
        pickle.dump(opp_adv_school_stats, f)
        f.close()


    # Convert to dataframes
    school_stats_df = pd.read_html(school_stats)
    adv_school_stats_df = pd.read_html(adv_school_stats)
    opp_school_stats_df = pd.read_html(opp_school_stats)
    adv_opp_school_stats_df = pd.read_html(opp_adv_school_stats)
    dataframes = [school_stats_df, adv_school_stats_df,opp_school_stats_df,adv_opp_school_stats_df]
    # Create a dataframe with ALL data
    full_school_stats = pd.DataFrame()
    # Start with school stats
    for i in range(0,len(dataframes)):
        for (first, sub) in dataframes[i][0]:
            # If it is the basic school stats, get the overall numbers
            if first == "Unnamed: 1_level_0" and i == 0 and sub == "School":
                column_data = dataframes[i][0][first][sub][dataframes[i][0][first][sub] != "School"].dropna()
                full_school_stats[sub] = column_data
                full_school_stats[sub] = full_school_stats[sub].str.upper()
                full_school_stats[sub] = full_school_stats[sub].str.replace("\\xa0NCAA", "")
                full_school_stats[sub] = full_school_stats[sub].str.replace(" ", "-")
                full_school_stats[sub] = full_school_stats[sub].str.replace("(", "")
                full_school_stats[sub] = full_school_stats[sub].str.replace(")", "")
                full_school_stats[sub] = full_school_stats[sub].str.replace("\\'", "")
                full_school_stats[sub] = full_school_stats[sub].str.replace("\\", "")
                full_school_stats[sub] = full_school_stats[sub].str.replace(".", "")
            elif first == "Overall" and i == 0:
                # Remove any rows that do not have a number
                column_data = dataframes[i][0][first][sub].fillna("0")
                column_data = column_data[column_data.apply(has_number)]
                full_school_stats[NAME_LOOKUP[sub]] = pd.to_numeric(column_data)
            elif first == "Points" and i == 0:
                # Remove any rows that do not have a number
                column_data = dataframes[i][0][first][sub].fillna("0")
                column_data = column_data[column_data.apply(has_number)]
                if sub == "Tm.":
                    full_school_stats["points"] = pd.to_numeric(column_data)
                else:
                    full_school_stats["opp_points"] = pd.to_numeric(column_data)
            elif first in ["Totals", "School Advanced"]:
                # Remove any rows that do not have a number
                column_data = dataframes[i][0][first][sub].fillna("0")
                column_data = column_data[column_data.apply(has_number)]
                full_school_stats[NAME_LOOKUP[sub]] = pd.to_numeric(column_data)
            elif first in ["Opponent","Opponent Advanced"]:
                # Remove any rows that do not have a number
                column_data = dataframes[i][0][first][sub].fillna("0")
                column_data = column_data[column_data.apply(has_number)]
                full_school_stats["opp_" + NAME_LOOKUP[sub]] = pd.to_numeric(column_data)

    # Add in implicit stats
    for i in range(0,2):
        if i == 0:
            opp_add = ""
        else:
            opp_add = "opp_"
        full_school_stats[opp_add + "defensive_rebounds"] = full_school_stats[opp_add + "total_rebounds"] - \
                                                            full_school_stats[opp_add + "offensive_rebounds"]
        full_school_stats[opp_add + "two_point_field_goals"] = full_school_stats[opp_add + "field_goals"] - \
                                                            full_school_stats[opp_add + "three_point_field_goals"]
        full_school_stats[opp_add + "two_point_field_goal_attempts"] = full_school_stats[opp_add+"field_goal_attempts"] - \
                                                            full_school_stats[opp_add+"three_point_field_goal_attempts"]
        full_school_stats[opp_add+"two_point_field_goal_percentage"] = full_school_stats[opp_add+"two_point_field_goals"] /\
                                                            full_school_stats[opp_add + "two_point_field_goal_attempts"]
    full_school_stats["games_played"] = full_school_stats["wins"] + full_school_stats["losses"]
    # Set school name as index
    full_school_stats.set_index("School", inplace=True)

    # Confirm that all stats are accounted for
    team_stats = ['assist_percentage', 'assists', 'block_percentage', 'blocks', 'defensive_rebounds',
                       'effective_field_goal_percentage', 'field_goal_attempts', 'field_goal_percentage',
                       'field_goals', 'free_throw_attempt_rate', 'free_throw_attempts', 'free_throw_percentage',
                       'free_throws', 'free_throws_per_field_goal_attempt', 'games_played', 'minutes_played',
                       'offensive_rating', 'offensive_rebound_percentage', 'offensive_rebounds',
                       'opp_assist_percentage', 'opp_assists', 'opp_block_percentage', 'opp_blocks',
                       'opp_defensive_rebounds', 'opp_effective_field_goal_percentage', 'opp_field_goal_attempts',
                       'opp_field_goal_percentage', 'opp_field_goals', 'opp_free_throw_attempt_rate',
                       'opp_free_throw_attempts', 'opp_free_throw_percentage', 'opp_free_throws',
                       'opp_free_throws_per_field_goal_attempt',
                       'opp_offensive_rebound_percentage', 'opp_offensive_rebounds', 'opp_personal_fouls',
                       'opp_points', 'opp_steal_percentage', 'opp_steals', 'opp_three_point_attempt_rate',
                       'opp_three_point_field_goal_attempts', 'opp_three_point_field_goal_percentage',
                       'opp_three_point_field_goals', 'opp_two_point_field_goal_attempts',
                       'opp_two_point_field_goal_percentage', 'opp_two_point_field_goals',
                       'opp_total_rebound_percentage', 'opp_total_rebounds', 'opp_true_shooting_percentage',
                       'opp_turnover_percentage', 'opp_turnovers', 'pace', 'personal_fouls', 'points',
                       'simple_rating_system', 'steal_percentage', 'steals', 'strength_of_schedule',
                       'three_point_attempt_rate', 'three_point_field_goal_attempts',
                       'three_point_field_goal_percentage', 'three_point_field_goals',
                       'two_point_field_goal_attempts', 'two_point_field_goal_percentage', 'two_point_field_goals',
                       'total_rebound_percentage', 'total_rebounds', 'true_shooting_percentage',
                       'turnover_percentage', 'turnovers', 'win_percentage']
    for s in team_stats:
        if s not in full_school_stats.columns:
            print(f"Missing {s}")
    return full_school_stats


if __name__ == "__main__":
    get_team_stats(2023)
