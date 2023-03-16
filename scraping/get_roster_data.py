import pickle
from scraping.smart_request import smart_request
import pandas as pd
import os

def get_url_name(name):
    if name == 'TCU':
        url_name = 'texas-christian'
    elif name == 'UAB':
        url_name = 'alabama-birmingham'
    elif name == 'UTSA':
        url_name = 'texas-san-antonio'
    elif name == 'UNC-ASHEVILLE':
        url_name = 'north-carolina-asheville'
    elif name == 'UNC-WILMINGTON':
        url_name = 'north-carolina-wilmington'
    elif name == 'UNC-GREENSBORO':
        url_name = 'north-carolina-greensboro'
    elif name == 'NC-STATE':
        url_name = 'north-carolina-state'
    elif name == "LOUISIANA":
        url_name = 'louisiana-lafayette'
    elif name == "UC-IRVINE":
        url_name = "california-irvine"
    elif name == "UC-DAVIS":
        url_name = 'california-davis'
    elif name == "UC-SANTA-BARBARA":
        url_name = 'california-santa-barbara'
    elif name == "LITTLE-ROCK":
        url_name = "arkansas-little-rock",
    elif name == "TCU":
        url_name = "texas-christian"
    else:
        url_name = name.lower().replace("(", "").replace(")", "").replace("&","")
    return url_name

def get_roster_stats(teams, year):
    """Gets the roster data for each team in the list
    Due to limits on requests, a sleep is put in place between requests

    Once data is collected, it saved to a pickle so that in the future, the request
    does not need to be made again"""
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "data")
    filename = os.path.join(full_path, f"roster_data_{year}.pckl")
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
            url_team = get_url_name(team)
            response = smart_request(f"https://www.sports-reference.com/cbb/schools/{url_team}/men/{year}.html")
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
    get_roster_stats(["Alabama", "Houston"], 2023)
