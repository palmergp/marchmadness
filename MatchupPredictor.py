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

    def __init__(self, model, features=None):
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
            else:
                self.model = pickle.load(f)
                self.explainer = None

                if "json" in features:
                    with open(features, "rb") as f:
                        self.features = json.load(f)
                else:
                    with open(features, "rb") as f:
                        self.features = pickle.load(f)
                # Check if naming convention used "team1/team2" instead of favorite/underdog
                self.features = [x.replace("Team1", "favorite_") for x in self.features]
                self.features = [x.replace("Team2", "underdog_") for x in self.features]

    def predict(self, data):
        result = self.model.predict(data)
        return result

    def main(self, year):
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
            #top5_total_per, top_per_percentage = get_per_stats(team1_roster)
            #sch_stats = get_ranked_stats(team1_schedule)
            #team1_data["top5_per_total"] = team1_roster["top5_per_total"]
            #team1_data["top_per_percentage"] = team1_roster["top_per_percentage"]
            for rost_stat in team1_roster.index:
                team1_data[rost_stat] = team1_roster[rost_stat]
            for sch_stat in team1_schedule.index:
                team1_data[sch_stat] = team1_schedule[sch_stat]
            #top5_total_per, top_per_percentage = get_per_stats(team2_roster)
            #sch_stats = get_ranked_stats(team2_schedule)
            #team2_data["top5_per_total"] = team1_roster["top5_per_total"]
            #team2_data["top_per_percentage"] = team1_roster["top_per_percentage"]
            for rost_stat in team2_roster.index:
                team2_data[rost_stat] = team2_roster[rost_stat]
            for sch_stat in team2_schedule.index:
                team2_data[sch_stat] = team2_schedule[sch_stat]

            # Get Bracket features
            predict_data = []
            for feat in self.features:
                if feat == "Round":
                    predict_data.append(round_num)
                elif feat == "favorite_seed" or "team1_seed":
                    predict_data.append(team1_seed)
                elif feat == "underdog_seed" or "team2_seed":
                    predict_data.append(team2_seed)
                elif feat == "SeedDiff":
                    predict_data.append(abs(team1_seed - team2_seed))
                else:
                    if "favorite" in feat:
                        predict_data.append(team1_data[feat.replace("favorite_","")])
                    else:
                        predict_data.append(team2_data[feat.replace("underdog_","")])

            # Make prediction
            print("Making prediction")
            winner_probs = self.model.predict_proba([predict_data])[0]
            # load feature names
            #with open(r"C:\Users\gppal\PycharmProjects\marchmadness\models\models23\v23_0_0\featurenames.pickle", "rb") as f:
            #    pickle.load(f)
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
            if self.explainer is not None:
                df = pd.DataFrame([predict_data], columns=self.features)
                df = df.astype("float64")  # final is the df of scaled features
                shap_values = self.explainer(df)
                plt.figure()  # plt is matplotlib
                f = shap.plots.waterfall(shap_values[0], show=False)
                f.set_title(f"Left ({team1_seed}) {team1}, Right ({team2_seed}) {team2}".title())
                plt.tight_layout()
                plt.show()
            print("-------------------------------------\n")
            print("Preparing for next prediction...\n")

if __name__ == '__main__':
    path = "models/models22/v3_1/"
    model = "Linear_SVC_v3_1.pickle"
    mp = MatchupPredictor(path+model, features=path+"featurenames.pickle")
    now = datetime.now()
    mp.main(now.year)
