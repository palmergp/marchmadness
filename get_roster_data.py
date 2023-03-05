import pickle
import requests
import pandas as pd


def get_2023_roster_stats(teams, filename="test_roster_data.pckl"):
    """Gets the roster data for each team in the list
    Due to limits on requests, a sleep is put in place between requests

    Once data is collected, it saved to a pickle so that in the future, the request
    does not need to be made again"""

    # First load the schedule data
    try:
        with open(filename, "rb") as f:
            roster_data = pickle.load(f)
    except FileNotFoundError:
        print("Could not find file. Assuming this is the first attempt at getting roster data")
        roster_data = pd.DataFrame()

    updated = False  # Indicates if we should save at the end
    # Loop through each team
    for team in teams:
        # Check if the team is in the saved data already
        already_collected = team in roster_data.index
        if not already_collected:
            # If we don't have it yet, then we need to collect all the stats for the team
            updated = True
            # Go grab the html
            print(f"Making request for {team} roster data")
            response = requests.get(f"https://www.sports-reference.com/cbb/schools/{team.lower()}/men/2023.html")
            team_roster = str(response.content)
            team_roster_df = pd.read_html(team_roster)[13]

            # Calculate the stats we care about
            team_row = {}
            team_row["School"] = team.upper()
            team_row["top5_per_total"] = team_roster_df['PER'].nlargest(5).sum()
            team_row["top_per_percentage"] = team_roster_df['PER'].max() / team_row["top5_per_total"]
            # Turn row into a dataframe and add to bigger dataframe
            team_row_df = pd.DataFrame(team_row, index=[0])
            team_row_df.set_index("School", inplace=True)
            roster_data = roster_data.append(team_row_df)

    if updated:
        # Save off changes
        with open(filename, "wb") as f:
            pickle.dump(roster_data,f)

    print("Done!!")
    return roster_data



if __name__ == "__main__":
    get_2023_roster_stats(["Alabama","Houston"],"test_roster_data.pckl")