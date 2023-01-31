# test_load.py
# Philip Oedi
# 2023-01-31

import os

import pandas as pd
import pytest

from daki_tools import load

# python -m pytest tests/


@pytest.fixture
def csv_data():
    file = os.path.join("data", "test.csv")
    return load.load(file)


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
    data = load.subset(csv_data, 1001)
    assert len(data) < len(csv_data)
