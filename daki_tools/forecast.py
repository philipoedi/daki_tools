# forecast.py
# Philip Oedi
# 2023-02-01

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class BaseForecast(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, fh, n):
        pass


class Ets(BaseForecast):
    def __init__(self, log_transform=True, **kwargs):
        self.model = ETSModel(np.empty(1))
        self.model_results = None
        self.forecast_start = None
        self.log_transform = log_transform
        if kwargs:
            for p, v in kwargs.items():
                setattr(self.model, p, v)

    def fit(self, data):
        """

        :param data:
        :return:
        """
        if self.log_transform:
            self.model.endog = np.log(data.value.values + 1e-6)
        else:
            self.model.endog = data.value.values
        self.model_results = self.model.fit(maxiter=10000, disp=0)
        self.forecast_start = data.index.max() + 1

    def predict(self, fh, n):
        y_pred = self.model_results.simulate(
            anchor="end", nsimulations=fh, repetitions=n
        )
        y_pred = pd.DataFrame(y_pred)
        y_pred["n"] = y_pred.index
        y_pred = pd.melt(y_pred, id_vars=["n"])
        y_pred["target"] = y_pred["n"].map(lambda x: x + self.forecast_start)
        y_pred = y_pred.rename({"variable": "sample_id"}, axis=1)
        y_pred = y_pred.drop("n", axis=1)
        if self.log_transform:
            y_pred.loc[:, "value"] = np.exp(y_pred["value"])
        return y_pred

    def fit_predict(self, data, fh, n):
        self.fit(data)
        return self.predict(fh, n)
