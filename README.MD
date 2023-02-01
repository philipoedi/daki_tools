# Daki Tools

Python code for [DAKI-FWS](https://github.com/rki-daki-fws/forecast-competition) baseline forecasting models. Can load training data for the forecasting challenge, generate forecasts (arima, ets) and transform the forecasts for the [submission requirements](https://github.com/rki-daki-fws/forecast-competition/blob/main/submissions/README.md).

## Overview

This repository has two modules `forecast` and `utils`. `utils` has functionality for loading and transforming training data from the challenge and for broadcasting the model fitting and forecast generation to multiple time-series. The `forecast` module specifies the `Ets` and `AutoArima` class to implement parameter estimation and simulation. It integrates `statsmodels` exponential smoothing and `pmdarima` auto-arima functionality into the pipeline that generates the forecasts. 

The package works only with data of the same format as provided in the forecast-competition.



* [statsmodels exponential smoothing](https://www.statsmodels.org/dev/examples/notebooks/generated/ets.html)
* [statsmodels arima](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html)
* [pmdarimas auto arima](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA)
## Installation

The package is best installed using conda. The package has been developed using `python 3.10.8`.

```
conda install --file requirements.txt
```

## Example

An example script can be found in the `example` folder.

To generate forecasts one needs to

1. Load the data (using `utils.load(file.csv)`)
2. Specify a model (e.g. `forecast.Ets(error="add")`)
3. Fit the model and simulate 

```python
from daki_tools import utils, forecast

# load the data
data = utils.load("data\\test.parquet")

# specify the model
model = forecast.Ets(error="add")
# model = Ets(error="add", trend="add") 
# model = Ets(error="add", trend="add", seasonal="add", seasonal_periods=7)
# model = AutoArima()

# fit and simulate
# for all locations (time-series) in the data
# will simulate 100 trajectories of 14-steps ahead forecasts
data_sim = utils.batch_forecast(data, model, fh=14, n=100)

```
