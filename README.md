# March Madness Predictor

## Overview
This repo contains all of the code used to generate and predict march madness brackets using machine learning

## Install
- Clone this repo
- Make a venv and activate it
- Run `pip install requirements.txt` (which I will totally add at some point)

# How to Use this Repo

## Creating a Model
Before starting the trainer process, you will first need to collect data. To collect data, run `collect_features.py`
in the scraping folder. If this is your first time running the trainer file, it will need to collect data and calculate features. This process
may take a long time. Data is collected via webscraping from sportsreference.com. They have rules in place to limit
the rate of requests that can be made which can slow down the process. Once data is collected, it is saved off for
future reference so no more requests need to be made. Once data is collected, the training process can be started.

To create a model, you will need to run the `Trainer.py` file.

Once data is collected, the trainer will begin training the different models. You can define how the trainer operates by
modifying the trainer config file in the configs folder. You have the following fields to edit:
- feature_list: file path to a JSON file listing what features should be used in the model
- data: file path to the data that will be used for training. This should be a pickle file output by `collect_features.py`
- version: string for the version number of the model. Used when saving the model
- outpath: filepath for where the model should be saved. A folder will be created here using the version number
- model_names: List of all models that should be created

# Predicting a Bracket
Also will add this... eventually

## Data Sources
Data is scrapped from the sportsreference.com page

# Features
This section details the various features used for a classification. There are 4 main types of features:
1. Basic Stats
2. Advanced Stats
3. Roster Stats
4. Schedule Stats

## Basic Stats
Basic stats are you standard metrics used to look at how a team performs. When collecting basic stats, the stats are <br>
collected for both the team and opposing teams against that team. Below is a list of all basic stat features:
- 