import numpy as np
from sklearn.linear_model import LinearRegression

def train_forecast(df):
    X = df['year'].values.reshape(-1, 1)
    y = df['co2'].values

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_future(model, start_year=2025, end_year=2050):
    years = np.arange(start_year, end_year).reshape(-1, 1)
    preds = model.predict(years)

    return years.flatten(), preds
