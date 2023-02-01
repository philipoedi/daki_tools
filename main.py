# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from daki_tools import forecast, utils

data = utils.load("data\\test.csv")

model = forecast.Ets(error="add", trend="add")
y_pred = utils.batch_forecast(data, model, fh=14, n=100)
y_pred_i = y_pred.query("location == 9162")

sns.lineplot(x=y_pred_i["target"].astype(int), y=y_pred_i["value"])
plt.show()

model = forecast.AutoArima(False)
y_pred = utils.batch_forecast(
    data.query("location == 9162 or location == 1001"), model, fh=14, n=100
)
y_pred_i = y_pred.query("location == 9162")

sns.lineplot(x=y_pred_i["target"].astype(int), y=y_pred_i["value"])
plt.show()
