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
                team1 = input("Team1 Name: ")
                team1_seed_true = int(input("Team1 Seed: "))
                try:
                    print("Fetching team stats...")
                    team1_data_true = all_teams.dataframes.loc[reformat_name(team1)]
                    team1_roster_true = all_teams[reformat_name(team1)].roster.players
                    team1_schedule_true = all_teams[reformat_name(team1)].schedule.dataframe
                    print("Successfully loaded 2022 stats for {}".format(team1))
                    not_loaded = False
                except KeyError:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(team1))
                    print(all_teams)
            # Get Team2 info
            not_loaded = True
            while not_loaded:
                team2 = input("Team2 Name: ")
                team2_seed_true = int(input("Team2 Seed: "))
                try:
                    print("Fetching team data...")
                    team2_data_true = all_teams.dataframes.loc[reformat_name(team2)]
                    team2_roster_true = all_teams[reformat_name(team2)].roster.players
                    team2_schedule_true = all_teams[reformat_name(team2)].schedule.dataframe
                    print("Successfully loaded 2022 stats for {}".format(team2))
                    not_loaded = False
                except KeyError:
                    print("Unable to load {}. Make sure it is spelled like it is in the following list:".format(team2))
                    print(all_teams)
            round_num = int(input("Round: "))
            print("Formatting data for prediction")

            # Get player and schedule stats
            top5_total_per, top_per_percentage = get_per_stats(team1_roster_true)
            sch_stats = get_ranked_stats(team1_schedule_true)
            team1_data_true["top5_total_per"] = top5_total_per
            team1_data_true["top_per_percentage"] = top_per_percentage
            for sch_stat in sch_stats:
                team1_data_true[sch_stat] = sch_stats[sch_stat]
            top5_total_per, top_per_percentage = get_per_stats(team2_roster_true)
            sch_stats = get_ranked_stats(team2_schedule_true)
            team2_data_true["top5_total_per"] = top5_total_per
            team2_data_true["top_per_percentage"] = top_per_percentage
            for sch_stat in sch_stats:
                team2_data_true[sch_stat] = sch_stats[sch_stat]

            # Run twice to reduce entry order bias
            total_probs = []
            for i in range(0, 2):
                # Flip so that each team is team1 and team2
                if i == 0:
                    team1_seed = team1_seed_true
                    team1_roster = team1_roster_true
                    team1_data = team1_data_true
                    team2_seed = team2_seed_true
                    team2_roster = team2_roster_true
                    team2_data = team2_data_true
                else:
                    team1_seed = team2_seed_true
                    # team1_roster = team2_roster_true
                    team1_data = team2_data_true
                    team2_seed = team1_seed_true
                    # team2_roster = team1_roster_true
                    team2_data = team1_data_true
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
                    # elif feat == "Team1top5_total_per":
                    #     top5_total_per, top_per_percentage = get_per_stats(team1_roster)
                    #     predict_data.append(top5_total_per)
                    # elif feat == "Team2top5_total_per":
                    #     top5_total_per, top_per_percentage = get_per_stats(team2_roster)
                    #     predict_data.append(top5_total_per)
                    # elif feat == "Team1top_per_percentage":
                    #     top5_total_per, top_per_percentage = get_per_stats(team1_roster)
                    #     predict_data.append(top_per_percentage)
                    # elif feat == "Team2top_per_percentage":
                    #     top5_total_per, top_per_percentage = get_per_stats(team2_roster)
                    #     predict_data.append(top_per_percentage)
                    else:
                        if "Team1" in feat:
                            predict_data.append(team1_data[feat.replace("Team1","")])
                        else:
                            predict_data.append(team2_data[feat.replace("Team2","")])

                # Make prediction
                print("Making prediction")
                winner_probs = self.model.predict_proba([predict_data])[0]
                total_probs.append(winner_probs)
            real_probs = [(total_probs[0][0] + total_probs[1][1]) / 2, (total_probs[0][1] + total_probs[1][0]) / 2]
            print("\n-------------------------------------")
            if real_probs[0] > real_probs[1]:
                print("The winner will be {}".format(team1))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, real_probs[0], team2, real_probs[1]))
            elif real_probs[0] <= real_probs[1]:
                print("The winner will be {}".format(team2))
                print("Probability split:\n\t{}: {}\n\t{}: {}".format(team1, real_probs[0], team2, real_probs[1]))
            else:
                print("Error: Returned value was unexpected!!")
            print("-------------------------------------\n")
            print("Preparing for next prediction...\n")

if __name__ == '__main__':
    path = "models/v3_1/"
    model = "Logistic_Regression_v3_1.pickle"
    feature_names = "featurenames.pickle"
    mp = MatchupPredictor(path+model, path+feature_names)
    mp.main()
