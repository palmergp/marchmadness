import pickle
from FeatureGenerator import reformat_name, get_per_stats, get_ranked_stats
import json
from scraping.get_team_data import get_team_stats
from scraping.get_schedule_data import get_schedule_stats
from scraping.get_roster_data import get_roster_stats
from sklearn import LinearSVC

class MatchupPredictor:

    def __init__(self, model, features):
        with open(model, "rb") as f:
            self.model = pickle.load(f)

        if "json" in features:
            with open(features, "rb") as f:
                self.features = json.load(f)
        else:
            with open(features, "rb") as f:
                self.features = pickle.load(f)

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
                except KeyError:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(first))
                    print(all_teams)
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
                except KeyError:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(second))
                    print(all_teams)
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
            print("{} {} is being used as the high seed and {} {}  is being used as the low seed".format(team2_seed,team2,team1_seed,team1))
            # Get player and schedule stats
            #top5_total_per, top_per_percentage = get_per_stats(team1_roster)
            #sch_stats = get_ranked_stats(team1_schedule)
            team1_data["top5_per_total"] = team1_roster["top5_per_total"]
            team1_data["top_per_percentage"] = team1_roster["top_per_percentage"]
            for sch_stat in team1_schedule.index:
                team1_data[sch_stat] = team1_schedule[sch_stat]
            #top5_total_per, top_per_percentage = get_per_stats(team2_roster)
            #sch_stats = get_ranked_stats(team2_schedule)
            team2_data["top5_per_total"] = team1_roster["top5_per_total"]
            team2_data["top_per_percentage"] = team1_roster["top_per_percentage"]
            for sch_stat in team2_schedule.index:
                team2_data[sch_stat] = team2_schedule[sch_stat]

            # Get Bracket features
            predict_data = []
            for feat in self.features:
                if feat == "Round":
                    predict_data.append(round_num)
                elif feat == "favorite_seed":
                    predict_data.append(team1_seed)
                elif feat == "underdog_seed":
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
            with open(r"C:\Users\gppal\PycharmProjects\marchmadness\models\models23\v23_0_0\featurenames.pickle", "rb") as f:
                pickle.load(f)
            clf = LinearSVC()
            
            print("\n-------------------------------------")
            if winner_probs[0] > winner_probs[1]:
                print("The winner will be {}".format(team1))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
            elif winner_probs[0] <= winner_probs[1]:
                print("The winner will be {}".format(team2))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, winner_probs[0], team2, winner_probs[1]))
            else:
                print("Error: Returned value was unexpected!!")
            print("-------------------------------------\n")
            print("Preparing for next prediction...\n")

if __name__ == '__main__':
    path = "models/models23/v23_0_0/"
    model = "Linear_SVC_v23_0_0.pickle"
    feature_names = "featurenames.pickle"
    mp = MatchupPredictor(path+model, path+feature_names)
    mp.main(2023)