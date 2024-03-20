# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from scraping.smart_request import smart_request
import os

# Define a custom function that takes a tag as an argument and returns True if the tag is a div and has
# a comment after it that says "game"
def find_game_div(tag):
  # Check if the tag is a div
  if tag.name == "div":
    # Get the next element of the tag (which should be either a newline or a comment)
    next_element = tag.next_element
    # Check if the next element is a comment and has the text "game"
    if next_element.next_sibling.strip() == "game":
      # Return True
      return True
  # Return False otherwise
  return False

# Define function
def get_tournament_data(year):
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "data")
    filename = os.path.join(full_path, str(year) + "_all_games.csv")
    try:
        df_bracket = pd.read_csv(filename)
    except FileNotFoundError:
        print("Tournament data not found. Pulling from sports reference...")
        # Get HTML content from url
        response = smart_request(f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html")
        soup = BeautifulSoup(response.text, "html.parser")

        # Find div with id brackets
        brackets = soup.find("div", id="brackets")

        df_bracket = pd.DataFrame()

        # Find all divs with ids east, midwest, south, west and national within brackets
        regions = brackets.find_all("div", id=["east", "midwest", "south", "west", "national"])

        # Loop through each region
        for region in regions:

            # Create an empty data frame for this region
            df_region = pd.DataFrame()

            # Find div with id bracket within this region
            bracket = region.find("div", id="bracket")

            # Find all divs with class round within this bracket
            rounds = bracket.find_all("div", class_="round")

            # Loop through each round
            if region['id'] == 'national':
                round_counter = 4
            else:
                round_counter = 0
            for round in rounds:
                round_counter = round_counter + 1
                # Find all divs containing team information within this round (every two divs are a matchup)
                games = round.find_all(find_game_div)

                # Loop through each game
                for game in games:
                    # Find winning team info (the one with class winner)
                    winner = game.find("div", class_="winner")
                    if winner is None:
                        # If there is no winner, then this is the champ of region
                        continue
                    winner_name = winner.find("a").text.strip()
                    winner_seed = winner.find("span").text.strip()
                    # Use re.search to find the first <a> tag that has a href attribute starting with "/cbb/boxscores"
                    # and get its text as the score
                    winner_score = re.search('<a href="/cbb/boxscores.*?>(.*?)</a>', str(winner)).group(1).strip()

                    # Find losing team info (the one without class)
                    loser = game.find("div", class_=None)
                    loser_name = loser.find("a").text.strip()
                    loser_seed = loser.find("span").text.strip()
                    # Use re.search to find the first <a> tag that has a href attribute starting with "/cbb/boxscores"
                    # and get its text as the score
                    loser_score = re.search('<a href="/cbb/boxscores.*?>(.*?)</a>', str(loser)).group(1).strip()

                    # Create a data frame for this matchup with columns: winning team, winning team seed,
                    # winning team score, losing team. losing team seed, and losing team score.
                    df_matchup = pd.DataFrame([[round_counter,
                                                winner_name,
                                                winner_seed,
                                                winner_score,
                                                loser_name,
                                                loser_seed,
                                                loser_score]],

                                              columns=["round",
                                                       "winning_team",
                                                       "winning_team_seed",
                                                       "winning_team_score",
                                                       "losing_team",
                                                       "losing_team_seed",
                                                       "losing_team_score"])

                    # Append this data frame to the region data frame
                    df_region = pd.concat([df_region, df_matchup])

            # Save the region data frame as a csv file with the region name as the file name
            df_bracket = df_bracket.append(df_region)

        df_bracket.to_csv(filename, index=False)

    return df_bracket


if __name__ == "__main__":
    for i in range(2010, 2011):
        if i != 2020:
            a = get_tournament_data(i)
