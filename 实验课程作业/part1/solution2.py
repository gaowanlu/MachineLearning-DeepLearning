import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

LINE_EXT_PATH = "./line-ext.csv"


def load_data(path):
    return pd.read_csv(path)


line_data = load_data(LINE_EXT_PATH)
# line_data.head()
# line_data.info()
x_train = line_data["YearsExperience"].values.reshape(-1, 1)
y_label = line_data["Salary"].values.reshape(-1, 1)

# create model instance
linearModel = LinearRegression()
# train fit
linearModel.fit(x_train, y_label)
print("w^ {} b^ {}".format(linearModel.coef_, linearModel.intercept_))
print("predict x=0.8452 y={} ", linearModel.predict(
    np.array([0.845]).reshape(-1, 1)))


# statsmodels
# statsmodels DOC https://devdocs.io/statsmodels/generated/statsmodels.regression.linear_model.ols

X = sm.add_constant(x_train)
statsmodels_model = sm.OLS(y_label, X)
results = statsmodels_model.fit()
print("use statsmodels params w^ {} b^ {}".format(
    results.params[1], results.params[0]))
