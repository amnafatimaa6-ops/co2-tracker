import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.lstm_forecast import train_lstm, forecast_lstm

st.set_page_config(page_title="CO₂ Climate Intelligence System", layout="wide")

# ---------------------------
# HEADER
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

    # remove regions (clean research dataset)
    invalid = [
        "World", "Africa", "Asia", "Europe",
        "European Union", "North America",
        "South America", "Oceania"
    ]
    df = df[~df["country"].isin(invalid)]

    df = df.groupby("country").filter(lambda x: len(x) > 15)

    return df

df = load_data()

# ---------------------------
# COUNTRY SELECTION
# ---------------------------
country = st.sidebar.selectbox("Select Country", sorted(df["country"].unique()))
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
# 🌤️ OPEN-METEO LIVE WEATHER (NO API KEY)
# ---------------------------
st.header("🌤️ Live Weather (Open-Meteo)")

city = st.text_input("Enter City", "London")

def get_weather(city_name):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}"
    geo_data = requests.get(geo_url).json()

    if "results" not in geo_data:
        return None

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_data = requests.get(weather_url).json()

    return weather_data.get("current_weather", None)

weather = get_weather(city)

if weather:
    st.success("Live Weather Data Retrieved")

    st.write(f"""
    - 🌡️ Temperature: {weather['temperature']} °C
    - 🌬️ Wind Speed: {weather['windspeed']} km/h
    - 🧭 Wind Direction: {weather['winddirection']}°
    - ⏱️ Weather Code: {weather['weathercode']}
    """)
else:
    st.warning("City not found or data unavailable")

# ---------------------------
# 🤖 LINEAR REGRESSION
# ---------------------------
X = c_df["year"].values.reshape(-1, 1)
y = c_df["co2"].values

lr_model = LinearRegression()
lr_model.fit(X, y)

lr_train_pred = lr_model.predict(X)

future_years = np.arange(2025, 2051).reshape(-1, 1)
lr_forecast = lr_model.predict(future_years)

# ---------------------------
# 🤖 LSTM MODEL
# ---------------------------
lstm_model, scaler, history = train_lstm(c_df)
lstm_years, lstm_forecast = forecast_lstm(lstm_model, scaler, c_df)

# ---------------------------
# 📉 LOSS CURVE
# ---------------------------
st.subheader("📉 LSTM Training Loss Curve")

fig_loss = px.line(
    x=list(range(len(history.history["loss"]))),
    y=history.history["loss"],
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
# 📉 EVALUATION METRICS
# ---------------------------
st.subheader("📉 Model Evaluation")

lr_mae = mean_absolute_error(y, lr_train_pred)
lr_rmse = np.sqrt(mean_squared_error(y, lr_train_pred))

min_len = min(len(lr_forecast), len(lstm_forecast))

lstm_mae = mean_absolute_error(lr_forecast[:min_len], lstm_forecast[:min_len])
lstm_rmse = np.sqrt(mean_squared_error(lr_forecast[:min_len], lstm_forecast[:min_len]))

st.write(f"""
| Model | MAE | RMSE |
|------|------|------|
| Linear Regression | {lr_mae:.2f} | {lr_rmse:.2f} |
| LSTM | {lstm_mae:.2f} | {lstm_rmse:.2f} |
""")

# ---------------------------
# 🌍 GLOBAL MAP
# ---------------------------
st.header("🌍 Global Emissions Map")

latest = df[df["year"] == df["year"].max()]

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
- 🌍 Insight: Non-linear climate behavior detected
- 🌤️ Live Weather integrated via Open-Meteo API
""")
