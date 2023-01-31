# load.py
# Philip Oedi
# 2023-01-31

import pandas as pd


def load(file) -> pd.DataFrame:
    """

    :param file:
    :return:
    """
    data = pd.read_csv(file)
    data["target"] = pd.to_datetime(data["target"], infer_datetime_format=True)
    data["target"] = pd.PeriodIndex(data["target"], freq="d")
    data.set_index("target", inplace=True)
    return data


def subset(data, location) -> pd.DataFrame:
    """

    :param data:
    :param location:
    :return:
    """
    return data.query(f"location == {location}").drop("location", axis=1)