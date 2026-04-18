import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data(df):
    df = df.sort_values("year")
    data = df["co2"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(5, len(scaled)):
        X.append(scaled[i-5:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, 5, 1)
    y = np.array(y)

    return X, y, scaler


def train_lstm(df):
    X, y, scaler = prepare_data(df)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(5, 1)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X, y,
        epochs=10,
        batch_size=16,
        verbose=0
    )

    return model, scaler, history


def forecast_lstm(model, scaler, df, steps=25):
    data = df["co2"].values.reshape(-1, 1)
    scaled = scaler.transform(data)

    seq = scaled[-5:].reshape(1, 5, 1)

    preds = []

    for _ in range(steps):
        p = model.predict(seq, verbose=0)[0][0]
        preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    years = np.arange(df["year"].max() + 1,
                      df["year"].max() + 1 + steps)

    return years, preds.flatten()
