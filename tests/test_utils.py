# test_utils.py
# Philip Oedi
# 2023-01-31

import os

import pandas as pd
import pytest

from daki_tools import forecast, utils

# python -m pytest tests/


@pytest.fixture
def csv_data():
    file = os.path.join("data", "test.csv")
    return utils.load(file)


def test_load_parquet():
    file = os.path.join("data", "test.parquet")
    assert isinstance(utils.load(file), pd.DataFrame)


def test_load_dataframe(csv_data):
    """

    :return:
    """
    assert isinstance(csv_data, pd.DataFrame)


def test_load_target_datatype(csv_data):
    """

    :return:
    """
    assert isinstance(csv_data.index, pd.PeriodIndex)


def test_subset(csv_data):
    """

    :param csv_data:
    :return:
    """
    data = utils.subset(csv_data, 1001)
    assert len(data) < len(csv_data)


def test_batch_forecast(csv_data):
    m = forecast.Ets(error="add")
    data = csv_data.query("location == 1001 or location == 1002")
    y_pred = utils.batch_forecast(data, m, 2, 2)
    # n_locations * fh * n
    assert y_pred.shape == (2 * 2 * 2, 4)


def test_batch_forecast_columns(csv_data):
    m = forecast.Ets(error="add")
    data = csv_data.query("location == 1001 or location == 1002")
    y_pred = utils.batch_forecast(data, m, 2, 2)
    # n_locations * fh * n
    assert y_pred.columns.to_list() == ["location", "target", "sample_id", "value"]
