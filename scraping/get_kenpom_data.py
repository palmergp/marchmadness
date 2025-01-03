from scraping.smart_request import smart_request
import pickle
import os
import pandas as pd


def get_kenpom_stats(year, force=False):
    """Loads the kenpom html and pulls out relevant features.
    The Kenpom site does not allow for webscraping so data must be pulled manually
    This can be done by going to the site in a browser and saving the webpage to scraping/data/kenpom_html

    If data has already been pulled in the past, it will be loaded from a pickle file
    """
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, "data")
    filename = os.path.join(full_path, f"kenpom_stats_{year}.pckl")
    # First load the schedule data
    try:
        if force:
            raise FileNotFoundError
        with open(filename, "rb") as f:
            kenpom_df = pickle.load(f)
    except FileNotFoundError:
        print("Could not find file. Assuming this is the first attempt at getting kenpom data")
        # Go grab the html
        print(f"Loading kenpom html from {year}")
        # with open(f"{full_path}/kenpom_html/{year} Pomeroy College Basketball Ratings.html", "r") as f:
        #    kenpom_html = f.read()
        kenpom_html = smart_request(f"https://kenpom.com/index.php?y={year}")
        # Convert to a dataframe
        kenpom_df = pd.read_html(kenpom_html)
        # Clean it up
        # columns = [c[1] for c in kenpom_df[0].columns]
        columns = ['kenpomRank', 'Team', 'Conf', 'W-L', 'NetRtg', 'ORtg', 'ORtg_rank', 'DRtg', 'DRtg_rank', 'AdjT', 'AdjT_rank',
                   'Luck', 'Luck_rank', 'AvgOppNetRtg', 'OppNetRtg_rank', 'AvgOppORtg', 'AvgOppORtg_rank', 'AvgOppDRtg',
                   'AvgOppDRtg_rank', 'NonConfNetRtg', 'NonConfNetRtg_rank']
        kenpom_df = pd.DataFrame(kenpom_df[0].values)
        kenpom_df.columns = columns
        # Remove the seed from the names
        kenpom_df["Team"] = kenpom_df["Team"].str.replace(r" [0-9]+", "", regex=True)
        # Reindex based on team
        kenpom_df.set_index("Team", inplace=True)
        # Remove unwanted columns
        droppable_columns = ['Conf', 'W-L'] + [c for c in columns if c.endswith('_rank')]
        kenpom_df.drop(columns=droppable_columns, inplace=True)

        # Save to a pickle file
        filename = os.path.join(full_path, f"kenpom_stats_{year}.pckl")
        with open(filename, "wb") as f:
            pickle.dump(kenpom_df, f)
    return kenpom_df


if __name__ == "__main__":
    get_kenpom_stats(2024, True)
