import pickle
from FeatureGenerator import reformat_name, get_per_stats, get_ranked_stats
from sportsipy.ncaab.teams import Team, Roster, Teams
import json

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

    def main(self):
        print("Starting 2022 March Madness Predictor!")
        print("Loading 2022 team data...")
        all_teams = Teams(2022)
        print("Team Data loaded")
        while True:
            # Get team 1 info
            not_loaded = True
            while not_loaded:
                first = input("Team1 Name: ")
                first_seed_true = int(input("Team1 Seed: "))
                try:
                    print("Fetching team stats...")
                    first_data_true = all_teams.dataframes.loc[reformat_name(first)]
                    first_roster_true = all_teams[reformat_name(first)].roster.players
                    first_schedule_true = all_teams[reformat_name(first)].schedule.dataframe
                    print("Successfully loaded 2022 stats for {}".format(first))
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
                    print("Fetching team data...")
                    second_data_true = all_teams.dataframes.loc[reformat_name(second)]
                    second_roster_true = all_teams[reformat_name(second)].roster.players
                    second_schedule_true = all_teams[reformat_name(second)].schedule.dataframe
                    print("Successfully loaded 2022 stats for {}".format(second))
                    not_loaded = False
                except KeyError:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(second))
                    print(all_teams)
            round_num = int(input("Round: "))
            print("Formatting data for prediction")

            # Make sure team 1 is the lower seed
            if first_seed_true >= second_seed_true:
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
            top5_total_per, top_per_percentage = get_per_stats(team1_roster)
            sch_stats = get_ranked_stats(team1_schedule)
            team1_data["top5_total_per"] = top5_total_per
            team1_data["top_per_percentage"] = top_per_percentage
            for sch_stat in sch_stats:
                team1_data[sch_stat] = sch_stats[sch_stat]
            top5_total_per, top_per_percentage = get_per_stats(team2_roster)
            sch_stats = get_ranked_stats(team2_schedule)
            team2_data["top5_total_per"] = top5_total_per
            team2_data["top_per_percentage"] = top_per_percentage
            for sch_stat in sch_stats:
                team2_data[sch_stat] = sch_stats[sch_stat]

            # Run twice to reduce entry order bias
            total_probs = []
            # Get Bracket features
            predict_data = []
            for feat in self.features:
                if feat == "Round":
                    predict_data.append(round_num)
                elif feat == "Team1Seed":
                    predict_data.append(team1_seed)
                elif feat == "Team2Seed":
                    predict_data.append(team2_seed)
                elif feat == "SeedDiff":
                    predict_data.append(abs(team1_seed - team2_seed))
                else:
                    if "Team1" in feat:
                        predict_data.append(team1_data[feat.replace("Team1","")])
                    else:
                        predict_data.append(team2_data[feat.replace("Team2","")])

            # Make prediction
            print("Making prediction")
            winner_probs = self.model.predict_proba([predict_data])[0]
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
    path = "models/v4_1/"
    model = "Logistic_Regression_v4_1.pickle"
    feature_names = "featurenames.pickle"
    mp = MatchupPredictor(path+model, path+feature_names)
    mp.main()
