from sklearn.linear_model import LinearRegression
from model.lstm_forecast import train_lstm, forecast_lstm

st.header("🔮 Model Comparison: ML vs Deep Learning")

# -------------------------
# LINEAR REGRESSION MODEL
# -------------------------
X = c_df['year'].values.reshape(-1, 1)
y = c_df['co2'].values

lr_model = LinearRegression()
lr_model.fit(X, y)

future_years = np.arange(2025, 2051).reshape(-1, 1)
lr_preds = lr_model.predict(future_years)

# -------------------------
# LSTM MODEL
# -------------------------
lstm_model, scaler = train_lstm(c_df)
lstm_years, lstm_preds = forecast_lstm(lstm_model, scaler, c_df)

# -------------------------
# CREATE DATAFRAMES
# -------------------------
lr_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "CO2": lr_preds,
    "Model": "Linear Regression"
})

lstm_df = pd.DataFrame({
    "Year": lstm_years,
    "CO2": lstm_preds,
    "Model": "LSTM (Deep Learning)"
})

combined = pd.concat([lr_df, lstm_df])

# -------------------------
# VISUAL COMPARISON
# -------------------------
fig = px.line(
    combined,
    x="Year",
    y="CO2",
    color="Model",
    title="Model Comparison: Linear Regression vs LSTM Forecast"
)

st.plotly_chart(fig, use_container_width=True)
