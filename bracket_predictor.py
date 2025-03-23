"""Given a year, uses a model to predict the bracket and calculate point total"""
import pandas as pd
from nonsense.favorite_picker import FavoritePicker
from matchup_predictor import MatchupPredictor
from scraping.get_tournament_data import get_tournament_data
import csv

class BracketPredictor:
    """Goes through a tournament and predicts all matchups"""

    def __init__(self, model, year):
        self.predictor = MatchupPredictor(model)
        self.set_year(year)

    def set_year(self, year):
        """Changes the year that is being predicted"""
        self.predictor.set_year(year)
        self.year = year
        self.bracket = get_tournament_data(year)
        # Account for a forfeit in 2021
        if self.year == 2021:
            new_row = pd.DataFrame(
                {
                    'winning_team': "Oregon",
                    'winning_team_seed': 7,
                    'losing_team': "VCU",
                    "losing_team_seed": 10,
                    "round": 1,
                    "winning_team_score": 0,
                    "losing_team_score": 0
                },
                index=[51]
            )
            self.bracket = pd.concat([self.bracket.iloc[:51], new_row, self.bracket.iloc[51:]]).reset_index(drop=True)
        # Reformat the bracket to be in the correct order
        new_order = list(range(0, 15)) + list(range(45, 60)) + list(range(15, 45)) + list(range(60, 63))
        # Reindex the DataFrame
        self.bracket = self.bracket.reindex(new_order).reset_index(drop=True)

    def main(self, tourney_over):
        """
        Makes a prediction for every matchup that happens in a tournament. Uses previous predictions to set up matchups
        in future rounds.
        :param tourney_over (bool) indicates if tournament is over and can be scored or not:
        :return:
            - total_points (int) - total points from bracket
            - picked_winner (bool) - boolean indicating if the winner was correctly predicted
        """
        # This needs to be able to pull in the round one matchups and figure out matchups from there
        # Cannot use anything after round 1 since missed picks affect future matchups
        finished_bracket = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        # Add the first round to the finished bracket
        for index, game in self.bracket[self.bracket["round"] == 1].iterrows():
            finished_bracket[1].append({
                "team": game["winning_team"],
                "seed": game["winning_team_seed"]
            })
            finished_bracket[1].append({
                "team": game["losing_team"],
                "seed": game["losing_team_seed"]
            })

        # once the first round is done, start doing the rest of matchups, even if they didnt actually happen
        for r in range(1, 7):
            for i in range(0, len(finished_bracket[r]), 2):
                result = self.predictor.predict(
                    first_team=finished_bracket[r][i]["team"],
                    first_seed=finished_bracket[r][i]["seed"],
                    second_team=finished_bracket[r][i+1]["team"],
                    second_seed=finished_bracket[r][i+1]["seed"],
                    round=r,
                    year=self.year
                )
                if result == 1:
                    finished_bracket[r+1].append({
                        "team": finished_bracket[r][i]["team"],
                        "seed": finished_bracket[r][i]["seed"]
                    })
                elif result == 2:
                    finished_bracket[r+1].append({
                        "team": finished_bracket[r][i+1]["team"],
                        "seed": finished_bracket[r][i+1]["seed"]
                    })
                else:
                    print(f"Error! Unable to predict {finished_bracket[r][i]['team']} vs {finished_bracket[r][i+1]['team']}")
                    raise Exception
        total_points = 0
        picked_winner = False
        if tourney_over:
            # Check how many points it would have gotten
            for r in range(2, 8):
                # Handle the special case for the champion
                if r == 7:
                    actual_teams = list(self.bracket[self.bracket["round"] == r-1]["winning_team"])
                else:
                    # Get all teams that made it to round 2
                    actual_teams = list(self.bracket[self.bracket["round"] == r]["winning_team"])
                    actual_teams = list(self.bracket[self.bracket["round"] == r]["losing_team"]) + actual_teams
                # Get all teams predicted in this round
                predicted_teams = [x["team"] for x in finished_bracket[r]]
                # Compare to see how many points
                total_points = total_points + len(set(actual_teams) & set(predicted_teams)) * 10 * (2**(r-2))
                # See if it got the winner right
                if r == 7 and actual_teams[0] == predicted_teams[0]:
                    picked_winner = True
            print(f"Model got {total_points} points")
        else:
            # Output the results in a file for input into pool
            # Combine "seed" and "team" into a single string and add "Round" header
            formatted_data = []
            for round_key, round_value in finished_bracket.items():
                formatted_column = [f"Round {round_key}"]
                formatted_column += [f"({item['seed']}) {item['team']}" for item in round_value]
                formatted_data.append(formatted_column)

            # Find the maximum length of the lists (to handle different lengths)
            max_length = max(len(column) for column in formatted_data)

            # Pad the shorter lists with empty strings
            for column in formatted_data:
                while len(column) < max_length:
                    column.append("")

            # Transpose the data to convert rows to columns
            transposed_data = list(zip(*formatted_data))

            # Write to CSV
            with open("output.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(transposed_data)
            print("Bracket saved!")
        return total_points, picked_winner


if __name__ == '__main__':
    version = "v25_7_0"
    path = f"models/models25/{version}/"
    # path = "nonsense/"
    model_pkg = f"KernelSVM_{version}.package"
    # model_pkg = "fav_picker.package"
    tourney_over = False
    bp = BracketPredictor(path+model_pkg, 2025)
    bp.main(tourney_over)
