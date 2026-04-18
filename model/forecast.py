from sklearn.linear_model import LinearRegression
import numpy as np

def train_forecast(country_df):
    X = country_df['year'].values.reshape(-1, 1)
    y = country_df['co2'].values

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_future(model, start_year=2025, end_year=2050):
    years = np.arange(start_year, end_year).reshape(-1, 1)
    predictions = model.predict(years)

    return years.flatten(), predictions
