from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import random

random.seed(11001)

def roc_plot_comparison(models, test_x, test_y):

    for m in models:
        yHat = models[m].predict_proba(test_x)
        pfa, pd, thresh = roc_curve(test_y, yHat[:1], pos_label=1)
        plt.plot(pfa, pd, label=m)
