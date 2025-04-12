import pickle
import random

from collect_features import reformat_name
from nonsense.favorite_picker import FavoritePicker
import json
from scraping.get_team_data import get_team_stats
from scraping.get_schedule_data import get_schedule_stats
from scraping.get_roster_data import get_roster_stats
from scraping.get_kenpom_data import get_kenpom_stats
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import pandas as pd

# pd.options.mode.copy_on_write = True


class MatchupPredictor:

    def __init__(self, model, features=None, show_plots=False):
        """Creates a Matchup Predictor by loading a model for predicting"""
        self.data = None
        self.show_plots=show_plots
        with open(model, "rb") as f:
            if "package" in model:
                package = pickle.load(f)
                self.model = package["model"]
                # From classifier in init
                self.explainer = None
                df = package["bg_dist_samp"]
                f = lambda x: self.model.predict_proba(x)[:, 1]
                self.explainer = shap.Explainer(f, df)
                self.features = package["feature_names"]
                if "scaler" in package:
                    self.scaler = package["scaler"]
                else:
                    self.scaler = None
            else:
                self.model = pickle.load(f)
                self.explainer = None
                self.scaler = None

                if "json" in features:
                    with open(features, "rb") as f:
                        self.features = json.load(f)
                else:
                    with open(features, "rb") as f:
                        self.features = pickle.load(f)
                # Check if naming convention used "team1/team2" instead of favorite/underdog
                self.features = [x.replace("Team1", "favorite_") for x in self.features]
                self.features = [x.replace("Team2", "underdog_") for x in self.features]

    def predict(self, first_team, first_seed, second_team, second_seed, round, year, same_seed=False):
        """Predicts a matchup
        Input:
            - first_team: (str) the name of the first team in the matchup
            - first_seed: (int) the seed of the first team in the matchup
            - second_team: (str) the name of the second team in the matchup
            - second_seed: (int) the seed of the second team in the matchup
            - round: (int) the round the matchup takes palce in
            - year: (int) the year in which the matchup takes place
            - show_plots: (bool) indicates whether plots should be created or not. Defaults to false
        Output:
         - result: (int) Indicates if team one or team two won. Output will either be 1 or 2. A -1 is
                    returned if an error occurs
        """
        try:
            # Try to load data
            print("Fetching team stats...")
            first_data = self.data.loc[reformat_name(first_team)]
            first_roster = get_roster_stats([reformat_name(first_team)], year).loc[reformat_name(first_team)]
            first_schedule = get_schedule_stats([reformat_name(first_team)], year).loc[reformat_name(first_team)]
            print("Successfully loaded {} stats for {}".format(year, first_team))
            second_data = self.data.loc[reformat_name(second_team)]
            second_roster = get_roster_stats([reformat_name(second_team)], year).loc[reformat_name(second_team)]
            second_schedule = get_schedule_stats([reformat_name(second_team)], year).loc[reformat_name(second_team)]
            print("Successfully loaded {} stats for {}".format(year, second_team))
        except KeyError as e:
            print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(first_team))
            print(self.data)
            print(e)
            result = -1
            return result
        # Make sure team 1 is the lower seed (better team)
        if first_seed < second_seed or same_seed:
            team1 = first_team
            team2 = second_team
            team1_seed = first_seed
            team1_roster = first_roster
            team1_data = first_data
            team1_schedule = first_schedule
            team2_seed = second_seed
            team2_roster = second_roster
            team2_data = second_data
            team2_schedule = second_schedule
        else:
            # If they are the same seed, run it twice and average the results
            if first_seed == second_seed:
                first_result, first_probs = self.predict(first_team, first_seed, second_team, second_seed, round, year, same_seed=True)
            team2 = first_team
            team1 = second_team
            team2_seed = first_seed
            team2_roster = first_roster
            team2_data = first_data
            team2_schedule = first_schedule
            team1_seed = second_seed
            team1_roster = second_roster
            team1_data = second_data
            team1_schedule = second_schedule
        print(
            "{} {} is being used as the underdog and {} {}  is being used as the favorite".format(team2_seed, team2,
                                                                                                  team1_seed,
                                                                                                  team1))
        # Get player and schedule stats
        team1_data = pd.concat([team1_data, team1_roster, team1_schedule])
        team2_data = pd.concat([team2_data, team2_roster, team2_schedule])
        # Get Bracket features
        predict_data = []
        for feat in self.features:
            if feat == "Round":
                predict_data.append(round)
            elif feat == "favorite_seed" or feat == "team1_seed":
                predict_data.append(team1_seed)
            elif feat == "underdog_seed" or feat == "team2_seed":
                predict_data.append(team2_seed)
            elif feat == "SeedDiff":
                predict_data.append(abs(team1_seed - team2_seed))
            else:
                if "favorite" in feat:
                    predict_data.append(team1_data[feat.replace("favorite_", "")])
                else:
                    predict_data.append(team2_data[feat.replace("underdog_", "")])
        # If it was a scaled model, scale the features
        if self.scaler:
            predict_data = self.scaler.transform([predict_data])[0]
        # Make prediction
        print("Making prediction")
        winner_probs = self.model.predict_proba([predict_data])[0]
        print("\n-------------------------------------")
        if winner_probs[0] > winner_probs[1]:
            winner_name = team1
            print("The winner will be {}".format(team1))
            print(
                "Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
        elif winner_probs[0] <= winner_probs[1]:
            winner_name = team2
            print("The winner will be {}".format(team2))
            print(
                "Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
        else:
            winner_name = ""
            print("Error: Returned value was unexpected!!")

        # Check if winner was input as team one or two
        if winner_name == first_team:
            result = 1
        elif winner_name == second_team:
            result = 2
        else:
            result = -1
        # From classifier in predict
        # Only show if upset
        if self.explainer is not None and winner_probs[0] <= winner_probs[1] and self.show_plots and not same_seed:
            df = pd.DataFrame([predict_data], columns=self.features)
            df = df.astype("float64")  # final is the df of scaled features
            shap_values = self.explainer(df)
            plt.figure()  # plt is matplotlib
            f = shap.plots.waterfall(shap_values[0], show=False)
            f.set_title(f"Left ({team1_seed}) {team1}, Right ({team2_seed}) {team2}".title())
            plt.tight_layout()
            plt.show()

        if same_seed:  # If same_seed is raised, then this is the recursive call
            return result, winner_probs
        elif not same_seed and team1_seed == team2_seed:
            # If same seed is not raised, but the two teams are the same seed, then this is the end
            # and the values should be averaged for a final result
            final_probs = [(winner_probs[0] + first_probs[1]) / 2, (winner_probs[1] + first_probs[0]) / 2]
            print("\n-------------------------------------")
            if final_probs[0] > final_probs[1]:
                winner_name = team1
                print("The winner will be {}".format(team1))
                print(
                    "Probability split:\n\t{}: {}\n\t{}: {}".format(team1, final_probs[0], team2, final_probs[1]))
            elif final_probs[0] <= final_probs[1]:
                winner_name = team2
                print("The winner will be {}".format(team2))
                print(
                    "Probability split:\n\t{}: {}\n\t{}: {}".format(team1, final_probs[0], team2, final_probs[1]))
            else:
                winner_name = ""
                print("Error: Returned value was unexpected!!")
            # Check if winner was input as team one or two
            if winner_name == first_team:
                result = 1
            elif winner_name == second_team:
                result = 2
            else:
                result = -1
        print("-------------------------------------\n")
        print("Preparing for next prediction...\n")
        return result

    def set_year(self, year):
        """Loads the data for a given year to be used for predictions"""
        self.data = get_team_stats(year)  #TODO: Use reformat names instead of whatever get_team_stats does
        # Fix UCF cause its dumb
        # self.data = self.data.rename(index={'UCF': 'CENTRAL-FLORIDA'})
        # Fix FDU too cause they are also dumb
        # self.data = self.data.rename(index={'FDU': 'FAIRLEIGH-DICKINSON'})
        self.data.index = self.data.index.map(reformat_name)

        # Add in Kenpom
        kenpom = get_kenpom_stats(year)
        kenpom.index = kenpom.index.map(reformat_name)
        indices_diff = self.data.index.difference(kenpom.index)
        self.data = pd.concat([
            self.data,
            kenpom
        ], axis=1)
        # Verify there are no NaNs as this would indicate the kenpom name didnt match up properly
        nan_rows = self.data[self.data['NetRtg'].isna()]
        if len(nan_rows) > 0:
            print("Some names are potentially off!!")
            print(nan_rows)

    def main(self):
        year = int(input("Enter the tournament year: "))
        self.set_year(year)
        while True:
            first_team = input("Team1 Name: ")
            first_seed = int(input("Team1 Seed: "))
            second_team = input("Team2 Name: ")
            second_seed = int(input("Team2 Seed: "))
            round = int(input("Round: "))
            result = self.predict(first_team, first_seed, second_team, second_seed, round, year)
            print(result)


if __name__ == '__main__':
    version = "v25_3_12"
    path = f"models/models25/{version}/"
    model_pkg = f"Neural_Network_{version}.package"
    mp = MatchupPredictor(path+model_pkg, features=path+"featurenames.pickle", show_plots=True)
    now = datetime.now()
    mp.main()
