#
# from statsmodels.tsa.exponential_smoothing.ets import ETSModel
#
#
#
# def forcast(data):
#     model = ETSModel(data.value.values)
#     fit = model.fit(maxiter=10000)
#     a = fit.simulate(anchor="end", nsimulations=14, repetitions=100)
#     k = pd.DataFrame(a)
#     k["n"] = k.index
#     pd.melt(k, id_vars=["n"])
