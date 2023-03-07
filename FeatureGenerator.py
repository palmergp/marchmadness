import csv
import pandas as pd
import pickle
import os
from sportsipy.ncaab.teams import Team, Roster, Teams

name_dict = {
    "UNC Asheville": "NORTH-CAROLINA-ASHEVILLE",
    "Texas A&M": "TEXAS-AM",
    "UNC Wilmington": "NORTH-CAROLINA-WILMINGTON",
    "UNC Greensboro": "NORTH-CAROLINA-GREENSBORO",
    "Middle Tennessee State": "MIDDLE-TENNESSEE",
    "Miami (Fla.)": "MIAMI-FL",
    "Stephen F. Austin": "STEPHEN-F-AUSTIN",
    "UC Berkeley": "CALIFORNIA",
    "St. Mary's (Cal.)": "SAINT-MARYS-CA",
    "saint marys": "SAINT-MARYS-CA",
    "Saint Marys": "SAINT-MARYS-CA",
    "Mount St. Mary's": "MOUNT-ST-MARYS",
    "Penn": "PENNSYLVANIA",
    "Long Island": "LONG-ISLAND-UNIVERSITY",
    "St. Bonaventure": "ST-BONAVENTURE",
    "TCU": "TEXAS-CHRISTIAN",
    "Loyola (Ill.)": "ILLINOIS-CHICAGO",
    "Murray St.": "MURRAY-STATE",
    "St. John's": "ST-JOHNS-NY",
    "LSU": "LOUISIANA-STATE",
    "lsu": "LOUISIANA-STATE",
    "UNLV": "NEVADA-LAS-VEGAS",
    "UTEP": "TEXAS-EL-PASO",
    "Detroit": "DETROIT-MERCY",
    "Loyola (Md.)": "LOYOLA-MD",
    "North Carolina A&T": "NORTH-CAROLINA-AT",
    "Albany (N.Y.)": "ALBANY-NY",
    "USC": "SOUTHERN-CALIFORNIA",
    "usc": "SOUTHERN-CALIFORNIA"
}


def reformat_name(name):
    """Fixes names with multiple spellings"""
    try:
        name = name_dict[name]
    except KeyError:
        name = name.replace(" ", "-")
        name = name.replace("St.", "SAINT")
        name = name.replace("'", "")
        name = name.replace("UC-", "CALIFORNIA-")
    return name.upper()


def get_per_stats(t_roster):
    # Get player based stats for team 1
    team_per = []
    for p in t_roster:
        try:
            if p.games_played > 15 and p.player_efficiency_rating:
                team_per.append(p.player_efficiency_rating)
        except TypeError:
            print("Skipping {} due to no data".format(p))
    # Sort least to greatest
    team_per = sorted(team_per)
    # Get top 5 PER total
    top5_total_per = sum(team_per[-5:])
    top_per_percentage = team_per[-1] / top5_total_per
    return top5_total_per, top_per_percentage

def get_ranked_stats(t_schedule):
    """Looks at a teams schedule and grabs basic stats from ranked games"""
    # Grab only ranked games
    t_schedule = t_schedule.loc[~t_schedule["opponent_rank"].isnull()]
    if not t_schedule.empty:
        # Calculate stats
        results = {}
        results["ranked_wins"] = len(t_schedule.loc[t_schedule["result"]=="Win"])
        results["ranked_losses"] = len(t_schedule.loc[t_schedule["result"]=="Loss"])
        results["ranked_win_percentage"] = results["ranked_wins"]/(results["ranked_wins"] + results["ranked_losses"])
        results["points_per_ranked"] = t_schedule["points_for"].sum()/len(t_schedule)
        results["opp_points_per_ranked"] = t_schedule["points_against"].sum()/len(t_schedule)
        results["margin_of_vict_ranked"] = results["points_per_ranked"] - results["opp_points_per_ranked"]
    else:
        results = {
            "ranked_wins": 0,
            "ranked_losses": 0,
            "ranked_win_percentage": 0,
            "points_per_ranked": 0,
            "opp_points_per_ranked":0,
            "margin_of_vict_ranked":0
        }
    return results


def rate_transform(data):
    # Divide all stats by minutes played to make them per minute
    nonfeatures = ["Year", "Name", "pace", "Team1", "Team2", "Winner", "Round", "strength_of_schedule"]
    for col in data.columns:
        if "rate" not in col and "percentage" not in col and "rating" not in col and col not in nonfeatures:
            if "Seed" in col:
                continue
            if "Team1" in col:
                data[col] = data[col] / data["Team1minutes_played"]
            else:
                data[col] = data[col] / data["Team2minutes_played"]
    return data

class FeatureGenerator:

    def __init__(self, data_paths, data=None, save_data=False):
        """Takes in multiple CSVs and generate features based on them"""
        self.data_paths = data_paths
        # TODO: Add features based on individual players
        self.team_stats = ['assist_percentage', 'assists', 'block_percentage', 'blocks', 'defensive_rebounds',
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
        self.player_stats = ['top5_per_total', 'top_per_percentage']
        self.schedule_stats = ['ranked_wins', 'ranked_losses', 'ranked_win_percentage', 'points_per_ranked',
                               'opp_points_per_ranked', 'margin_of_vict_ranked']
        df = {"Year": [],
             "Team1": [],
             "Team2": [],
             "Winner": [],
             "Round": [],
             "Team1Seed": [],
             "Team2Seed": [],
             "SeedDiff": []}
        for stat in self.team_stats:
            df["Team1" + stat] = []
            df["Team2" + stat] = []
        for stat in self.player_stats:
            df["Team1" + stat] = []
            df["Team2" + stat] = []
        for stat in self.schedule_stats:
            df["Team1" + stat] = []
            df["Team2" + stat] = []
        self.data = pd.DataFrame(df)
        self.data["Year"] = pd.Series([], dtype=object)
        self.label_names = ["Team1", "Team2"]
        self.labels = []
        self.feature_names = []
        self.features = []
        if data:
            print("Loading games from file")
            file = open(data, 'rb')
            self.data = pickle.load(file)
            file.close()
        else:
            # Load csvs
            for file in data_paths:
                with open(file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    isHeader = True
                    flip = False  # To make sure team 1 isnt always the winner
                    if "data_cleaned.csv" in file:
                        for row in csv_reader:
                            if isHeader:
                                header = row
                                isHeader = False
                            else:
                                if flip:
                                    df_row = {"Winner": "Team1"}
                                else:
                                    df_row = {"Winner": "Team2"}
                                for i in range(0,len(header)):
                                    if header[i] == "YEAR":
                                        df_row["Year"] = int(row[i])
                                    elif header[i] == "ROUND":
                                        df_row["Round"] = int(row[i])
                                    elif header[i] == "WSEED":
                                        if flip:
                                            df_row["Team1Seed"] = int(row[i])
                                        else:
                                            df_row["Team2Seed"] = int(row[i])
                                    elif header[i] == "WTEAM":
                                        if flip:
                                            df_row["Team1"] = row[i]
                                        else:
                                            df_row["Team2"] = row[i]
                                    elif header[i] == "LSEED":
                                        if flip:
                                            df_row["Team2Seed"] = int(row[i])
                                        else:
                                            df_row["Team1Seed"] = int(row[i])
                                    elif header[i] == "LTEAM":
                                        if flip:
                                            df_row["Team2"] = row[i]
                                        else:
                                            df_row["Team1"] = row[i]
                                for stat in self.team_stats:
                                    df_row["Team1" + stat] = -1
                                    df_row["Team2" + stat] = -1
                                flip = not flip
                                df_row["SeedDiff"] = abs(df_row["Team1Seed"] - df_row["Team2Seed"])
                                self.data = self.data.append(df_row, ignore_index=True)

            # Currently can only get data back to 2010 so drop all other years
            self.data = self.data.loc[self.data["Year"] >= 2010]

            if save_data:
                file = open("./data/games_2010on.pickle",'wb')
                pickle.dump(self.data, file)
                file.close()
                print("New data base saved")


    def run(self, save_name=None):
        """Generates features and outputs a csv"""
        # Do one year at a time
        skip_years = []
        years = list(set(self.data["Year"]))
        for year in years:
            if year in skip_years:
                print("Skipping {}".format(year))
                continue
            print("Processing year {}".format(year))
            # Pull all teams from that year
            all_teams = self.get_year_data(year)
            # Get list of unique teams for that year
            team_names = set(list(set(self.data.loc[self.data['Year'] == year]["Team1"])) + list(set(self.data.loc[self.data['Year']==year]["Team2"])))
            for name in team_names:
                print("\tProcessing {} {}".format(year, name))
                # Get stats for each team and apply to all applicable entries in data
                formatted_name = reformat_name(name)
                try:
                    team_data = all_teams.dataframes.loc[formatted_name]
                    for stat in self.team_stats:
                        self.data.loc[(self.data["Year"] == year) & (self.data["Team1"] == name), "Team1" + stat] = \
                            team_data[stat]
                        self.data.loc[(self.data["Year"] == year) & (self.data["Team2"] == name), "Team2" + stat] = \
                            team_data[stat]

                    # Get player based stats for team 1
                    t_roster = all_teams[formatted_name].roster.players
                    self.get_player_data(t_roster, name, year)
                    t_schedule = all_teams[formatted_name].schedule.dataframe
                    self.get_schedule_data(t_schedule, name, year)


                except TypeError as e:
                    try:
                        # Try polling Team directly
                        print("Trying to poll directly for {} {}".format(name, year))
                        team_obj = Team(formatted_name, year=year)
                        team_data = team_obj.dataframe
                        for stat in self.team_stats:
                            self.data.loc[(self.data["Year"] == year) & (self.data["Team1"] == name), "Team1" + stat] = \
                            list(team_data[stat])[0]
                            self.data.loc[(self.data["Year"] == year) & (self.data["Team2"] == name), "Team2" + stat] = \
                            list(team_data[stat])[0]
                        t_roster = team_obj.roster.players
                        self.get_player_data(t_roster, name, year)
                        t_schedule = team_obj.schedule.dataframe
                        self.get_schedule_data(t_schedule, name, year)
                    except TypeError as e2:
                        print(e2)
                        print("No data for {} {}".format(year, name))
                        continue



        # Save data
        if save_name:
            outpath = "featuresets/{}".format(save_name)
            try:
                os.mkdir(outpath)
            except FileExistsError:
                pass
            file = open('{}/feature_data_{}.pickle'.format(outpath,save_name), 'wb')
            pickle.dump(self.data, file)
            file = open('{}/features_{}.pickle'.format(outpath,save_name), 'wb')
            pickle.dump(self.feature_names, file)

    def get_player_data(self, t_roster, name, year):
        top5_total_per, top_per_percentage = get_per_stats(t_roster)
        self.data.loc[(self.data["Year"] == year) & (self.data["Team1"] == name), "Team1top5_total_per"] = \
            top5_total_per
        self.data.loc[(self.data["Year"] == year) & (self.data["Team2"] == name), "Team2top5_total_per"] = \
            top5_total_per
        self.data.loc[(self.data["Year"] == year) & (self.data["Team1"] == name), "Team1top_per_percentage"] = \
            top_per_percentage
        self.data.loc[(self.data["Year"] == year) & (self.data["Team2"] == name), "Team2top_per_percentage"] = \
            top_per_percentage

    def get_schedule_data(self, t_schedule, name, year):
        results = get_ranked_stats(t_schedule)
        for s_stat in self.schedule_stats:
            self.data.loc[(self.data["Year"] == year) & (self.data["Team1"] == name), "Team1" + s_stat] = \
                results[s_stat]
            self.data.loc[(self.data["Year"] == year) & (self.data["Team2"] == name), "Team2" + s_stat] = \
                results[s_stat]

    def get_year_data(self, year):
        try:
            file = open('./data/years/{}.pickle'.format(year), 'rb')
            all_teams = pickle.load(file)
            file.close()
            print("Loaded {} teams from pickle file".format(year))
        except FileNotFoundError:
            print("Pickle file for {} does not exist. Pulling from internet...".format(year))
            all_teams = Teams(year)
            # Save off as pickle file
            file = open('./data/years/{}.pickle'.format(year), 'wb')
            pickle.dump(all_teams, file)
        return all_teams


if __name__ == '__main__':
    data_folder = "./data/"
    d = [data_folder + "data_cleaned.csv"]
    fg = FeatureGenerator(d, data="./data/games_2010on.pickle")
    fg.run("v3_0")
