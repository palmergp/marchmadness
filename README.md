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
For now, you need to manually go through each individual matchup and fill our your bracket. Eventually, I would like to
make it so that you can upload a single file and it will predict the full bracket on its own. We'll see if I ever get to
that.

## Data Sources
Data is scrapped from the sportsreference.com page

# Features
This section details the various features used for a classification. There are 4 main types of features:
1. Basic Stats
2. Advanced Stats
3. Roster Stats
4. Schedule Stats
5. Kenpom Stats

## Basic Stats
Basic stats are you standard metrics used to look at how a team performs. When collecting basic stats, the stats are
collected for both the team and opposing teams against that team. Below is a list of all basic stat features:

## Advanced Stats
Advanced stats come from the advanced metrics table on the team page. These stats are collected for both the team and
opponent teams

## Roster Stats
Roster stats look at the individual players on the team

## Schedule Stats
Schedule stats look at the teams that the team has played, specifically ranked teams, to determine how they perform
against other good teams

### Kenpom Stats
Kenpom stats are advanced metrics pulled from kenpom.com

## Full list of features
For each feature, there is a version for both the favorite and the underdog team
- seed: The seed the team has going into the tournament
- win_percentage: The teams win percentage during the regular season
- simple_rating_system: The team's simple rating system value
- strength_of_schedule: The strength of schedule during the regular season
- points: Total points scored during the regular season
- opp_points: Total points allowed during the regular season
- field_goals: Total number of field goals made in the regular season
- field_goal_attempts: Total number of field goals attempted during the regular season
- field_goal_percentage: Field goal percentage during the season
- three_point_field_goals: Total number of 3 pointers attempted during the season
- three_point_field_goal_attempts: Total number of 3 pointers attempted
- three_point_field_goal_percentage
- free_throws
- free_throw_attempts
- free_throw_percentage
- offensive_rebounds
- total_rebounds
- assists
- steals
- blocks
- turnovers
- personal_fouls
- pace
- offensive_rating
- free_throw_attempt_rate
- three_point_attempt_rate
- true_shooting_percentage
- total_rebound_percentage
- assist_percentage
- steal_percentage
- block_percentage
- effective_field_goal_percentage
- turnover_percentage
- offensive_rebound_percentage
- free_throws_per_field_goal_attempt
- opp_minutes_played
- opp_field_goals
- opp_field_goal_attempts
- opp_field_goal_percentage
- opp_three_point_field_goals
- opp_three_point_field_goal_attempts
- opp_three_point_field_goal_percentage
- opp_free_throws
- opp_free_throw_attempts
- opp_free_throw_percentage
- opp_offensive_rebounds
- opp_total_rebounds
- opp_assists
- opp_steals
- opp_blocks
- opp_turnovers
- opp_personal_fouls
- opp_pace
- opp_offensive_rating
- opp_free_throw_attempt_rate
- opp_three_point_attempt_rate
- opp_true_shooting_percentage
- opp_total_rebound_percentage
- opp_assist_percentage
- opp_steal_percentage
- opp_block_percentage
- opp_effective_field_goal_percentage
- opp_turnover_percentage
- opp_offensive_rebound_percentage
- opp_free_throws_per_field_goal_attempt
- defensive_rebounds
- two_point_field_goals
- two_point_field_goal_attempts
- two_point_field_goal_percentage
- opp_defensive_rebounds
- opp_two_point_field_goals
- opp_two_point_field_goal_attempts
- opp_two_point_field_goal_percentage
- games_played
- ranked_wins
- ranked_losses
- ranked_win_percentage
- points_per_ranked
- opp_points_per_ranked
- margin_of_vict_ranked
- top5_per_total
- top_per_percentage
- seed_diff
- last10_win_percentage: Win percentage in final 10 games before tournament
- kenpomRank: Rank given to the team by kenpom
- NetRtg
- ORtg
- DRtg
- AdjT
- Luck
- AvgOppNetRtg
- AvgOppORtg
- AvgOppDRtg 
- NonConfNetRtg

## Adding a new feature
If you are interested in adding a new feature, here are the steps you should take to bring it into the models
1. Go to the scraping function (get_whatever.py) that has the data needed for the feature
2. Add the logic that will collect the new feature
3. Delete old pickle files that contain data from the associated data set 
4. Rerun collect_features.py
5. Add the new feature to the feature list
6. Retrain model