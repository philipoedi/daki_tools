# utils.py
# Philip Oedi
# 2023-01-31

import pandas as pd


def load(file) -> pd.DataFrame:
    """
    loads csv or parquet file from the forecast-competition.

    :param file: `str`
    :return: `pandas.DataFrame`
    """
    if ".csv" in file:
        data = pd.read_csv(file)
    elif ".parquet" in file:
        data = pd.read_parquet(file)
    else:
        print("Choose parquet or csv file")
    data["target"] = pd.to_datetime(data["target"], infer_datetime_format=True)
    data["target"] = pd.PeriodIndex(data["target"], freq="d")
    data.set_index("target", inplace=True)
    return data


def subset(data, location) -> pd.DataFrame:
    """
    Creates a single time-series for a specified location.

    :param data: `pandas.DataFrame` with columns `location`, `value` and the index `target`, the reporting date
    :param location: `int` county id
    :return: `pandas.DataFrame`
    """
    return data.query(f"location == {location}").drop("location", axis=1)


def batch_forecast(data, model, fh, n):
    """
    Fits and simulates forecasts for a specified model for all locations in the data.
    The simulations are concatenated to a `pandas.DataFrame` that matches the forecast-competition
    submission requirements https://github.com/rki-daki-fws/forecast-competition/blob/main/submissions/README.md

    :param data: `pandas.DataFrame` with columns `location`, `value` and index `target`
    :param model: `forecast.Ets` or `forecast.AutoArima`
    :param fh: `int` forecasting horizon
    :param n: `int` number of samples/repititions
    :return: `pandas.DataFrame`
    """

    def fn(x):
        y_pred = model.fit_predict(subset(data, x), fh, n)
        y_pred["location"] = x
        return y_pred

    y_pred = pd.concat(map(fn, pd.unique(data.location)))
    y_pred = y_pred[["location", "target", "sample_id", "value"]]
    return y_pred
