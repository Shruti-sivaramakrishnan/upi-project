import pandas as pd, numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("processed/upi_monthly.csv", parse_dates=["month"]).sort_values("month")
y = df.set_index("month")["volume_mn"].asfreq("MS")
train = y.iloc[:60]
print("Training window:", train.index[0].date(), "to", train.index[-1].date(), "n =", len(train))

fits = {}
for name, (o, s) in {"Model 1": ((1,1,1),(1,1,1,12)), "Model 2": ((0,1,1),(0,1,1,12))}.items():
    fits[name] = SARIMAX(train, order=o, seasonal_order=s).fit(disp=False)
    print(name, "AIC", round(fits[name].aic, 1), "BIC", round(fits[name].bic, 1))
print("Expected: Model 1 AIC 684.0 BIC 693.3 | Model 2 AIC 681.0 BIC 686.6")

r = fits["Model 2"]
tab = pd.DataFrame({"coef": r.params, "std_err": r.bse, "p_value": r.pvalues}).round(4)
print(tab.to_string())
print("Invertible (|MA coef| < 1):", bool((r.params.filter(like="ma").abs() < 1).all()))
tab.to_csv("processed/sarima_model2_coefficients.csv")
