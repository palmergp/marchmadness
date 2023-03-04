import pickle
import requests
import pandas as pd


def get_2023_schedule_stats(teams, filename="test_schedule_data.pckl"):
    """Gets the schedule data for each team in the list
    Due to limits on requests, a sleep is put in place between requests

    Once data is collected, it saved to a pickle so that in the future, the request
    does not need to be made again"""

    # First load the schedule data
    try:
        with open(filename, "rb") as f:
            schedule_data = pickle.load(f)
    except FileNotFoundError:
        print("Could not find file. Assuming this is the first attempt at getting schedule data")
        schedule_data = pd.DataFrame()

    updated = False  # Indicates if we should save at the end
    # Loop through each team
    for team in teams:
        # Check if the team is in the saved data already
        already_collected = team in schedule_data
        if not already_collected:
            # If we don't have it yet, then we need to collect all the stats for the team
            updated = True
            # Go grab the html
            response = requests.get(f"https://www.sports-reference.com/cbb/schools/{team.lower()}/men/2023-schedule.html")
            team_schedule = str(response.content)
            team_schedule_df = pd.read_html(team_schedule)[1]
            # We really only care about ranked games so remove all games against unranked opponents
            team_schedule_df = team_schedule_df[team_schedule_df['Opponent'].str.contains('\d')]
            # Calculate the stats we care about
            team_row = {}
            team_row["School"] = team
            team_row["ranked_wins"] = team_schedule_df['Unnamed: 8'].value_counts()['W']
            team_row["ranked_losses"] = team_schedule_df['Unnamed: 8'].value_counts()['L']
            ranked_games = team_row["ranked_wins"]+team_row["ranked_losses"]
            team_row["ranked_win_percentage"] = team_row["ranked_wins"]/ranked_games
            team_row["points_per_ranked"] = team_schedule_df['Tm'].sum()/ranked_games
            team_row["opp_points_per_ranked"] = team_schedule_df['Opp'].sum()/ranked_games
            team_row["margin_of_vict_ranked"] = team_row["points_per_ranked"] - team_row["opp_points_per_ranked"]
            # Turn row into a dataframe and add to bigger dataframe
            team_row_df = pd.DataFrame(team_row, index=[0])
            schedule_data = schedule_data.append(team_row_df)
    schedule_data.set_index("School", inplace=True)

    if updated:
        # Save off changes
        with open(filename, "wb") as f:
            pickle.dump(schedule_data,f)

    print("Done!!")



if __name__ == "__main__":
    get_2023_schedule_stats(["Alabama","Houston"],"test_schedule_data.pckl")