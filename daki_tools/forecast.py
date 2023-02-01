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
    def __init__(self, **kwargs):
        self.model = ETSModel(np.empty(1))
        self.model_results = None
        if kwargs:
            for p, v in kwargs.items():
                setattr(self.model, p, v)

    def fit(self, data):
        """

        :param data:
        :return:
        """
        self.model.endog = data.value.values
        self.model_results = self.model.fit(maxiter=10000)

    def predict(self, fh, n):
        y_pred = self.model_results.simulate(
            anchor="end", nsimulations=fh, repetitions=n
        )
        y_pred = pd.DataFrame(y_pred)
        y_pred["n"] = y_pred.index
        return pd.melt(y_pred, id_vars=["n"])

    def fit_predict(self, data, fh, n):
        self.fit(data)
        return self.predict(fh, n)
