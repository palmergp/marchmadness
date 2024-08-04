"""Given a year, uses a model to predict the bracket and calculate point total"""
from matchup_predictor import MatchupPredictor
from scraping.get_tournament_data import get_tournament_data


class BracketPredictor:
    """Goes through a tournament and predicts all matchups"""

    def __init__(self, model, year):
        self.predictor = MatchupPredictor(model)
        self.bracket = get_tournament_data(year)
        self.year = year

    def main(self):
        # This needs to be able to pull in the round one matchups and figure out matchups from there
        # Cannot use anything after round 1 since missed picks affect future matchups
        points = 0
        for game in self.bracket:
            result = self.predictor.predict(
                first_team=game["team1"],
                first_seed=game["team1_seed"],
                second_team=game["team2"],
                second_seed=game["team2_seed"],
                round=game["round"],
                year=self.year,
            )
            if tourney_over:
                # If the tourney is over, check the result to see if the prediction was right
                pass

if __name__ == '__main__':
    path = "models/models24/v24_3_1/"
    model = "Linear_SVC_v24_3_1.package"
    year = 2024
    tourney_over = True
    bp = BracketPredictor(path+model,2024)
    bp.main(tourney_over)