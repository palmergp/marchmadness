import pickle
import requests
import pandas as pd
from get_roster_data import get_url_name


def get_schedule_stats(teams, year):
    """Gets the schedule data for each team in the list
    Due to limits on requests, a sleep is put in place between requests

    Once data is collected, it saved to a pickle so that in the future, the request
    does not need to be made again"""

    filename = f"schedule_data_{year}.pckl"
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
        already_collected = team in schedule_data.index
        if not already_collected:
            # If we don't have it yet, then we need to collect all the stats for the team
            updated = True
            # Go grab the html
            print(f"Making requests for {team} schedule data")
            url_team = get_url_name(team)
            response = requests.get(f"https://www.sports-reference.com/cbb/schools/{url_team}/men/{year}-schedule.html")
            team_schedule = str(response.content)
            team_schedule_df = pd.read_html(team_schedule)[1]
            # Fill nan opponents
            team_schedule_df["Opponent"] = team_schedule_df["Opponent"].fillna("None")
            # We really only care about ranked games so remove all games against unranked opponents
            team_schedule_df = team_schedule_df[team_schedule_df['Opponent'].str.contains('\d')]
            team_schedule_df = team_schedule_df[team_schedule_df['Type']!="NCAA"]
            # Calculate the stats we care about
            team_row = {}
            team_row["School"] = team.upper()
            if team_schedule_df.empty:
                # If it is empty, mark everything as zero
                team_row["ranked_wins"] = 0
                team_row["ranked_losses"] = 0
                team_row["ranked_win_percentage"] = 0
                team_row["points_per_ranked"] = 0
                team_row["opp_points_per_ranked"] = 0
                team_row["margin_of_vict_ranked"] = 0
            else:
                ranked_results = team_schedule_df['Unnamed: 8'].value_counts()
                if 'W' in ranked_results:
                    team_row["ranked_wins"] = ranked_results['W']
                else:
                    team_row["ranked_wins"] = 0
                if 'L' in ranked_results:
                    team_row["ranked_losses"] = ranked_results['L']
                else:
                    team_row["ranked_losses"] = 0
                ranked_games = team_row["ranked_wins"]+team_row["ranked_losses"]
                if ranked_games > 0:
                    team_row["ranked_win_percentage"] = team_row["ranked_wins"]/ranked_games
                    team_row["points_per_ranked"] = pd.to_numeric(team_schedule_df['Tm']).sum()/ranked_games
                    team_row["opp_points_per_ranked"] = pd.to_numeric(team_schedule_df['Opp']).sum()/ranked_games
                    team_row["margin_of_vict_ranked"] = team_row["points_per_ranked"] - team_row["opp_points_per_ranked"]
                else:
                    team_row["ranked_win_percentage"] = 0
                    team_row["points_per_ranked"] = 0
                    team_row["opp_points_per_ranked"] = 0
                    team_row["margin_of_vict_ranked"] = 0
            # Turn row into a dataframe and add to bigger dataframe
            team_row_df = pd.DataFrame(team_row, index=[0])
            team_row_df.set_index("School", inplace=True)
            schedule_data = schedule_data.append(team_row_df)


    if updated:
        # Save off changes
        with open(filename, "wb") as f:
            pickle.dump(schedule_data,f)

    print("Done!!")
    return schedule_data


if __name__ == "__main__":
    get_schedule_stats(["michigan"], 2021)
