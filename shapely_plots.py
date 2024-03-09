##################
#  shapley plots #
##################
import shap
from shutil import register_unpack_format
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import logging

def create_shap_global_plots(shap_values, output_path, current_model=""):
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    fig.tight_layout()
    try:
        plt.savefig(output_path + current_model +"_shap_global_bar.png")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR: {e}")

    fig = plt.figure()
    shap.plots.beeswarm(shap_values,show=False)
    fig.tight_layout()
    try:
        plt.savefig(output_path + current_model + "_shap_global_beeswarm_.png")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR: {e}")

    fig = plt.figure()
    shap.summary_plot(shap_values, plot_type='violin',show=False)
    fig.tight_layout()
    try:
        plt.savefig(output_path + current_model + "_shap_global_violin_.png")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR: {e}")

    fig = plt.figure()
    shap.plots.heatmap(shap_values)
    fig.tight_layout()
    try:
        plt.savefig(output_path + current_model + "_shap_global_heatmap_.png")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR: {e}")


def create_shap_local_plots(shap_values, path,current_model=""):
    #Waterfall
    fig = plt.figure()
    shap.plots.waterfall(shap_values, show=False)
    fig.tight_layout()
    try:
        plt.savefig(path + current_model + "_shap_local_waterfall_malicious_.png")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR: {e}")
    # Kinda the same as above

def create_shap_explainer(model,X):
    X = X.sample(frac=1)
    try:
        explainer = shap.explainers.Permutation(model.predict_proba,X)
    except Exception as e:
        print(f"Error: {e}")

def shap_preprocessing(explainer,x,type="all"):
    max_sample=10000
    if len(x) < max_sample:
        try:
            shap_value_all = explainer(x[:len(x)])
        except Exception as e:
            print(f"Error:{e}")

    if type == 1: # malicious
        shap_values_1 = shap_value_all[...,1]
        return shap_values_1
    elif type == 0:  # non-malicious
        shap_values_0 = shap_value_all[...,0]
        return shap_values_0
    elif type == "all":
        return shap_value_all