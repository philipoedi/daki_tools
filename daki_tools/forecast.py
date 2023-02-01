# forecast.py
# Philip Oedi
# 2023-02-01

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class BaseForecast(ABC):
    def __init__(self):
        # date for which the first forecast will be generated
        self.forecast_start = None

    @abstractmethod
    def fit(self, data):
        """
        estimate parameters of the specified model given data

        :param data: `pandas.DataFrame` with index target(date) and column value (r-value or incidence)
        """

    @abstractmethod
    def predict(self, fh, n) -> pd.DataFrame:
        """
        simulates future trajectories from the fit model for a specified time horizon. Takes the day after the most
        recent day in the training data as a starting date.

        :param fh: `int` how many steps to forecast
        :param n: `int` repetitions/samples. `n = 2` will generate two trajectories
        :return: `pandas.DataFrame` with columns `target`, `sample_id` and `value`
        """

    def fit_predict(self, data, fh, n) -> pd.DataFrame:
        self.fit(data)
        return self.predict(fh, n)

    def set_params(self, **kwargs):
        """
        parse model parameters to the model

        :param kwargs: any parameter as found in the base models from `statsmodels` can be added
        """
        if kwargs:
            for p, v in kwargs.items():
                setattr(self.model, p, v)

    def transform_predictions(self, y_pred) -> pd.DataFrame:
        """
        Transform the simulation output to a `pandas.DataFrame` that can be concatanated with others
        to fit the submission requirements. Adds the date of the forecast data, makes data long format
        and renames columns

        :param y_pred: `numpy.ndarray` of shape `(fh, n), a column representing a simulated trajectory
        :return: `pandas.DataFrame` in long format with columns `sample_id`, `value` and `target`
        """
        y_pred = pd.DataFrame(y_pred)
        y_pred["n"] = y_pred.index
        y_pred = pd.melt(y_pred, id_vars=["n"])
        y_pred["target"] = y_pred["n"].map(lambda x: x + self.forecast_start)
        y_pred = y_pred.rename({"variable": "sample_id"}, axis=1)
        y_pred = y_pred.drop("n", axis=1)
        if self.log_transform:
            y_pred.loc[:, "value"] = np.exp(y_pred["value"])
        return y_pred


class Ets(BaseForecast):
    def __init__(self, log_transform=True, **kwargs):
        """

        :param log_transform: `bool`
        :param kwargs: any parameter from `statsmodels.tsa.exponential_smoothing.ets.ETSModel`
        """
        self.model = ETSModel(np.empty(1))  # initial model
        self.model_results = None
        self.log_transform = log_transform
        self.set_params(**kwargs)

    def fit(self, data):
        """
        fits the ets model to the data. Applies log-transformation if specified in the object

        :param data: `pandas.DataFrame` with index target(date) and column value (r-value or incidence)
        """
        if self.log_transform:
            self.model.endog = np.log(data.value.values + 1e-6)
        else:
            self.model.endog = data.value.values
        self.model_results = self.model.fit(maxiter=10000, disp=0)
        self.forecast_start = data.index.max() + 1

    def predict(self, fh, n) -> pd.DataFrame:
        """
        generate forecasts from the Ets model and transforms to daki-format

        :param fh: `int` how many steps to forecast
        :param n: `int` repetitions/samples. `n = 2` will generate two trajectories
        :return: `pandas.DataFrame` with columns `target`, `sample_id` and `value`
        """
        y_pred = self.model_results.simulate(
            anchor="end", nsimulations=fh, repetitions=n
        )
        return self.transform_predictions(y_pred)


class AutoArima(BaseForecast):
    def __init__(self, log_transform=True):
        self.model = None
        self.log_transform = True

    def fit(self, data):
        """
        fits the arima model to the data. Applies log-transformation if specified in the object
        :param data: `pandas.DataFrame` with index target(date) and column value (r-value or incidence)
        """
        if self.log_transform:
            y = np.log(data.value.values + 1e-6)
        else:
            y = data.value.values
        self.model = pm.auto_arima(y)
        self.forecast_start = data.index.max() + 1

    def predict(self, fh, n) -> pd.DataFrame:
        """
        generate forecasts from the arima model and transforms to daki-format.

        :param fh: `int` how many steps to forecast
        :param n: `int` repetitions/samples. `n = 2` will generate two trajectories
        :return: `pandas.DataFrame` with columns `target`, `sample_id` and `value`
        """
        y_pred = self.model.arima_res_.simulate(
            anchor="end", nsimulations=fh, repetitions=n
        )
        return self.transform_predictions(y_pred.squeeze(axis=1))
