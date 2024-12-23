import pickle
from collect_features import reformat_name
import json
from scraping.get_team_data import get_team_stats
from scraping.get_schedule_data import get_schedule_stats
from scraping.get_roster_data import get_roster_stats
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import pandas as pd


class MatchupPredictor:

    def __init__(self, early_model, late_model=None, round_split=2, features=None):
        """Used to predict matchups
        Input:
            - early_model: Model used for games earlier in the tournament
            - late_model: Model used for games later in the tournament. If left blank, the early model
                            will be used for all games
            - round_split: Round at which the late model begins to be used.
                            Example: If round_split=2, the early model will be used for round 1 and
                            the late model will be used for round 2+
        """
        self.round_split = round_split
        with open(early_model, "rb") as f:
            package = pickle.load(f)
            self.early_model = package["model"]
            # From classifier in init
            self.early_explainer = None
            df = package["bg_dist_samp"]
            f = lambda x: self.early_model.predict_proba(x)[:, 1]
            self.early_explainer = shap.Explainer(f, df)
            self.features = package["feature_names"]
            if "scaler" in package:
                self.early_scaler = package["scaler"]
            else:
                self.early_scaler = None
        if late_model is None:
            # If a late model wasnt provided, use the early model as both
            self.late_model = self.early_model
            self.late_explainer = self.early_explainer
            self.late_scaler = self.early_scaler
        else:
            with open(late_model, "rb") as f:
                package = pickle.load(f)
                self.late_model = package["model"]
                # From classifier in init
                self.late_explainer = None
                df = package["bg_dist_samp"]
                f = lambda x: self.late_model.predict_proba(x)[:, 1]
                self.late_explainer = shap.Explainer(f, df)
                self.features = package["feature_names"]
                if "scaler" in package:
                    self.late_scaler = package["scaler"]
                else:
                    self.late_scaler = None

    def main(self, year):
        """Prompts the user for matchup info and then makes a prediction"""
        print(f"Starting {year} March Madness Predictor!")
        print(f"Loading {year} team data...")
        all_teams = get_team_stats(year)
        print("Team Data loaded")
        while True:
            # Get team 1 info
            not_loaded = True
            while not_loaded:
                first = input("Team1 Name: ")
                first_seed_true = int(input("Team1 Seed: "))
                try:
                    print("Fetching team stats...")
                    first_data_true = all_teams.loc[reformat_name(first)]
                    first_roster_true = get_roster_stats([reformat_name(first)], year).loc[reformat_name(first)]
                    first_schedule_true = get_schedule_stats([reformat_name(first)], year).loc[reformat_name(first)]
                    print("Successfully loaded {} stats for {}".format(year, first))
                    not_loaded = False
                except KeyError as e:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(first))
                    print(all_teams)
                    print(e)
            # Get Team2 info
            not_loaded = True
            while not_loaded:
                second = input("Team2 Name: ")
                second_seed_true = int(input("Team2 Seed: "))
                try:
                    second_data_true = all_teams.loc[reformat_name(second)]
                    second_roster_true = get_roster_stats([reformat_name(second)], year).loc[reformat_name(second)]
                    second_schedule_true = get_schedule_stats([reformat_name(second)], year).loc[reformat_name(second)]
                    print("Successfully loaded {} stats for {}".format(year, second))
                    not_loaded = False
                except KeyError as e:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(second))
                    print(all_teams)
                    print(e)
            round_num = int(input("Round: "))
            print("Formatting data for prediction")

            # Make sure team 1 is the lower seed
            if first_seed_true <= second_seed_true:
                team1 = first
                team2 = second
                team1_seed = first_seed_true
                team1_roster = first_roster_true
                team1_data = first_data_true
                team1_schedule = first_schedule_true
                team2_seed = second_seed_true
                team2_roster = second_roster_true
                team2_data = second_data_true
                team2_schedule = second_schedule_true
            else:
                team2=first
                team1=second
                team2_seed = first_seed_true
                team2_roster = first_roster_true
                team2_data = first_data_true
                team2_schedule = first_schedule_true
                team1_seed = second_seed_true
                team1_roster = second_roster_true
                team1_data = second_data_true
                team1_schedule = second_schedule_true
            print("{} {} is being used as the underdog and {} {}  is being used as the favorite".format(team2_seed,team2,team1_seed,team1))
            # Get player and schedule stats
            for rost_stat in team1_roster.index:
                team1_data[rost_stat] = team1_roster[rost_stat]
            for sch_stat in team1_schedule.index:
                team1_data[sch_stat] = team1_schedule[sch_stat]
            for rost_stat in team2_roster.index:
                team2_data[rost_stat] = team2_roster[rost_stat]
            for sch_stat in team2_schedule.index:
                team2_data[sch_stat] = team2_schedule[sch_stat]

            # Get Bracket features
            predict_data = []
            for feat in self.features:
                if feat == "Round":
                    predict_data.append(round_num)
                elif feat == "favorite_seed" or feat == "team1_seed":
                    predict_data.append(team1_seed)
                elif feat == "underdog_seed" or feat == "team2_seed":
                    predict_data.append(team2_seed)
                elif feat == "SeedDiff":
                    predict_data.append(abs(team1_seed - team2_seed))
                else:
                    if "favorite" in feat:
                        predict_data.append(team1_data[feat.replace("favorite_","")])
                    else:
                        predict_data.append(team2_data[feat.replace("underdog_","")])

            # Choose which classifier to use based on the round
            if round_num >= self.round_split:
                scaler = self.late_scaler
                model = self.late_model
                explainer = self.late_explainer
            else:
                scaler = self.early_scaler
                model = self.early_model
                explainer = self.early_explainer

            # If it was a scaled model, scale the features
            if scaler:
                predict_data = scaler.transform([predict_data])[0]
            # Make prediction
            print("Making prediction")
            winner_probs = model.predict_proba([predict_data])[0]
            print("\n-------------------------------------")
            if winner_probs[0] > winner_probs[1]:
                print("The winner will be {}".format(team1))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
            elif winner_probs[0] <= winner_probs[1]:
                print("The winner will be {}".format(team2))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
            else:
                print("Error: Returned value was unexpected!!")
            # From classifier in predict
            # Only show if upset
            if explainer is not None and winner_probs[0] <= winner_probs[1]:
                df = pd.DataFrame([predict_data], columns=self.features)
                df = df.astype("float64")  # final is the df of scaled features
                shap_values = explainer(df)
                plt.figure()  # plt is matplotlib
                f = shap.plots.waterfall(shap_values[0], show=False)
                f.set_title(f"Left ({team1_seed}) {team1}, Right ({team2_seed}) {team2}".title())
                plt.tight_layout()
                plt.show()
            print("-------------------------------------\n")
            print("Preparing for next prediction...\n")

if __name__ == '__main__':
    path = "models/models24/v24_4_0/"
    early_model = "early_Logistic_Regression_v24_4_0.package"
    late_model = "late_Logistic_Regression_v24_4_0.package"
    mp = MatchupPredictor(
        early_model=path+early_model,
        late_model=path+late_model)
    now = datetime.now()
    mp.main(now.year)
