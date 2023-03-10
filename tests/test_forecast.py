# test_foreacast.py
# Philip Oedi
# 2023-02-01

import os

import pytest
from pmdarima.arima import arima
from statsmodels.tsa.exponential_smoothing.ets import ETSResultsWrapper

from daki_tools import forecast, utils


@pytest.fixture
def csv_data():
    file = os.path.join("data", "test.csv")
    data = utils.load(file)
    return utils.subset(data, 1001)


def test_ets_fit(csv_data):
    """

    :param csv_data:
    :return:
    """
    model = forecast.Ets()
    model.fit(csv_data)
    assert isinstance(model.model_results, ETSResultsWrapper)


def test_ets_predict(csv_data):
    """

    :param csv_data:
    :return:
    """
    model = forecast.Ets()
    model.fit(csv_data)
    y_pred = model.predict(fh=14, n=10)
    assert y_pred.shape == (14 * 10, 3)


def test_ets_init():
    model = forecast.Ets(error="add")
    assert model.model.error == "add"


def test_ets_fit_predict(csv_data):
    ets_aa = forecast.Ets(error="add", trend="add")
    y_pred = ets_aa.fit_predict(csv_data, 14, 10)
    assert y_pred.shape == (14 * 10, 3)


def test_ets_fit_predict_outcome(csv_data):
    ets_aa = forecast.Ets(error="add", trend="add")
    y_pred = ets_aa.fit_predict(csv_data, 14, 10)
    assert y_pred.shape == (14 * 10, 3)


def test_ets_fit_predict_outcome_log(csv_data):
    ets_aa = forecast.Ets(error="add", trend="add", log_transform=False)
    y_pred = ets_aa.fit_predict(csv_data, 14, 10)
    assert ets_aa.log_transform == False


def test_arima_fit(csv_data):
    m = forecast.AutoArima()
    m.fit(csv_data)
    assert isinstance(m.model, arima.ARIMA)


def test_arima_fit_predict(csv_data):
    m = forecast.AutoArima()
    y_pred = m.fit_predict(csv_data, fh=4, n=10)
    assert y_pred.shape == (4 * 10, 3)
