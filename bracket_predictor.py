"""Given a year, uses a model to predict the bracket and calculate point total"""
import pandas as pd

from matchup_predictor import MatchupPredictor
from scraping.get_tournament_data import get_tournament_data

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
        new_order = list(range(0, 15)) + list(range(45, 60)) + list(range(15, 45)) + list(range(60, 62))
        # Reindex the DataFrame
        self.bracket = self.bracket.reindex(new_order).reset_index(drop=True)

    def main(self, tourney_over):
        # This needs to be able to pull in the round one matchups and figure out matchups from there
        # Cannot use anything after round 1 since missed picks affect future matchups
        finished_bracket = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
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
        for r in range(1, 6):
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
        if tourney_over:
            total_points = 0
            # Check how many points it would have gotten
            for r in range(2, 7):
                # Get all teams that made it to round 2
                actual_teams = list(self.bracket[self.bracket["round"] == r]["winning_team"])
                actual_teams = list(self.bracket[self.bracket["round"] == r]["losing_team"]) + actual_teams
                # Get all teams predicted in this round
                predicted_teams = [x["team"] for x in finished_bracket[r]]
                # Compare to see how many points
                total_points = total_points + len(set(actual_teams) & set(predicted_teams)) * 10 * (r-1)
            print(f"Model got {total_points} points")
        return total_points


if __name__ == '__main__':
    path = "models/models24/v24_3_0/"
    model_pkg = "Linear_SVC_v24_3_0.package"
    tourney_over = True
    bp = BracketPredictor(path+model_pkg, 2021)
    bp.main(tourney_over)