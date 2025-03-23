from scraping.smart_request import smart_request
import pickle
import os
import pandas as pd


archive_link = {
    2024: "https://web.archive.org/web/20240320182234/https://kenpom.com/",
    2023: "https://web.archive.org/web/20230315101006/https://kenpom.com/",
    2022: "https://web.archive.org/web/20220316200732/https://kenpom.com/",
    2021: "https://web.archive.org/web/20210317104513/https://kenpom.com/",
    2019: "https://web.archive.org/web/20190320205526/https://kenpom.com/",
    2018: "https://web.archive.org/web/20180314201532/https://kenpom.com/",
    2017: "https://web.archive.org/web/20170315023549/https://kenpom.com/",
    2016: "https://web.archive.org/web/20160315071332/http://kenpom.com/",
    2015: "https://web.archive.org/web/20150318080652/http://kenpom.com/",
    2014: "https://web.archive.org/web/20140319015649/http://kenpom.com/",
    2013: "https://web.archive.org/web/20130320132059/http://kenpom.com/",
    2012: "https://web.archive.org/web/20120311165019/http://kenpom.com/",
    2011: "https://web.archive.org/web/20110311233233/http://kenpom.com/"
}

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
        if year in archive_link:
            kenpom_html = smart_request(archive_link[year])
        else:
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

        if year <= 2016:
            kenpom_df["NetRtg"] = kenpom_df["ORtg"] - kenpom_df["DRtg"]
            kenpom_df["AvgOppNetRtg"] = kenpom_df["AvgOppORtg"] - kenpom_df["AvgOppDRtg"]

        # Save to a pickle file
        filename = os.path.join(full_path, f"kenpom_stats_{year}.pckl")
        with open(filename, "wb") as f:
            pickle.dump(kenpom_df, f)
    return kenpom_df


if __name__ == "__main__":
    for year in range(2011,2025):
        if year == 2020:
            continue
        get_kenpom_stats(year, True)
