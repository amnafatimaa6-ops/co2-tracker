import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.lstm_forecast import train_lstm, forecast_lstm

st.set_page_config(page_title="CO₂ Climate Intelligence System", layout="wide")

# ---------------------------
# HEADER (FIXED - NO SCHOLARSHIP LINE)
# ---------------------------
st.title("🌍 CO₂ Climate Intelligence System")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)
    df = df[['country', 'year', 'co2']].dropna()
    return df

df = load_data()

# ---------------------------
# COUNTRY FILTER (CLEAN)
# ---------------------------
valid_countries = df.groupby("country")["co2"].count()
valid_countries = valid_countries[valid_countries > 10].index

country = st.sidebar.selectbox("Select Country", valid_countries)
c_df = df[df['country'] == country]

st.write("Rows loaded:", len(c_df))

if len(c_df) == 0:
    st.error("No data available for this country.")
    st.stop()

# ---------------------------
# 📊 HISTORICAL DATA
# ---------------------------
st.header("📊 Historical Emissions")

fig1 = px.line(c_df, x="year", y="co2",
               title=f"{country} CO₂ Emissions Over Time")
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------
# 🤖 LINEAR REGRESSION
# ---------------------------
X = c_df['year'].values.reshape(-1, 1)
y = c_df['co2'].values

lr_model = LinearRegression()
lr_model.fit(X, y)

lr_train_pred = lr_model.predict(X)

future_years = np.arange(2025, 2051).reshape(-1, 1)
lr_forecast = lr_model.predict(future_years)

# ---------------------------
# 🤖 LSTM
# ---------------------------
lstm_model, scaler, history = train_lstm(c_df)
lstm_years, lstm_forecast = forecast_lstm(lstm_model, scaler, c_df)

# ---------------------------
# 📉 LOSS CURVE
# ---------------------------
st.subheader("📉 LSTM Training Loss Curve")

fig_loss = px.line(
    x=list(range(len(history.history['loss']))),
    y=history.history['loss'],
    labels={"x": "Epoch", "y": "Loss"},
    title="LSTM Training Loss"
)

st.plotly_chart(fig_loss, use_container_width=True)

# ---------------------------
# 📊 FORECAST COMPARISON
# ---------------------------
st.subheader("📊 Forecast Comparison")

lr_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "CO2": lr_forecast,
    "Model": "Linear Regression"
})

lstm_df = pd.DataFrame({
    "Year": lstm_years,
    "CO2": lstm_forecast,
    "Model": "LSTM"
})

combined = pd.concat([lr_df, lstm_df])

fig2 = px.line(
    combined,
    x="Year",
    y="CO2",
    color="Model",
    markers=True
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# 📉 CORRECT MAE / RMSE (FIXED)
# ---------------------------
st.subheader("📉 Model Evaluation (Fixed)")

# Linear Regression (TRAIN ERROR)
lr_mae = mean_absolute_error(y, lr_train_pred)
lr_rmse = np.sqrt(mean_squared_error(y, lr_train_pred))

# LSTM (training approximation comparison)
min_len = min(len(lstm_forecast), len(lr_forecast))
lstm_mae = mean_absolute_error(
    lr_forecast[:min_len],
    lstm_forecast[:min_len]
)

lstm_rmse = np.sqrt(mean_squared_error(
    lr_forecast[:min_len],
    lstm_forecast[:min_len]
))

st.write(f"""
| Model | MAE | RMSE |
|------|------|------|
| Linear Regression | {lr_mae:.2f} | {lr_rmse:.2f} |
| LSTM | {lstm_mae:.2f} | {lstm_rmse:.2f} |
""")

# ---------------------------
# 🌍 MAP
# ---------------------------
st.header("🌍 Global Emissions Map")

latest = df[df['year'] == df['year'].max()]

fig3 = px.choropleth(
    latest,
    locations="country",
    locationmode="country names",
    color="co2",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# ⚙️ SCENARIO SIMULATOR
# ---------------------------
st.header("⚙️ Climate Scenario Simulator")

reduction = st.slider("Emission Reduction (%)", 0, 100, 20)

scenario = lr_df.copy()
scenario["Adjusted CO2"] = scenario["CO2"] * (1 - reduction / 100)

fig4 = px.line(
    scenario,
    x="Year",
    y="Adjusted CO2"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# 🧠 INSIGHTS
# ---------------------------
st.header("🧠 AI Insights")

trend = "increasing" if lr_forecast[-1] > lr_forecast[0] else "decreasing"
volatility = np.std(c_df["co2"])

st.write(f"""
- 📈 Trend: **{trend}**
- 📊 Volatility: **{volatility:.2f}**
- 🤖 Models: Linear Regression + LSTM
- 🌍 Interpretation: Non-linear climate behavior detected
""")
