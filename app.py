import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.lstm_forecast import train_lstm, forecast_lstm

st.set_page_config(page_title="CO₂ Climate Intelligence System", layout="wide")

st.title("🌍 CO₂ Climate Intelligence System")
st.write("Scholarship-grade AI climate analytics platform")

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
# COUNTRY SELECT
# ---------------------------
country = st.sidebar.selectbox("Select Country", df['country'].unique())
c_df = df[df['country'] == country]

# ---------------------------
# 📊 HISTORICAL DATA
# ---------------------------
st.header("📊 Historical Emissions")

fig1 = px.line(c_df, x="year", y="co2",
               title=f"{country} CO₂ Emissions Over Time")
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------
# 🤖 MODELS
# ---------------------------
st.header("🔮 Forecasting Models Comparison")

# Linear Regression
X = c_df['year'].values.reshape(-1, 1)
y = c_df['co2'].values

lr_model = LinearRegression()
lr_model.fit(X, y)

future_years = np.arange(2025, 2051).reshape(-1, 1)
lr_pred = lr_model.predict(future_years)

# LSTM
lstm_model, scaler = train_lstm(c_df)
lstm_years, lstm_pred = forecast_lstm(lstm_model, scaler, c_df)

# ---------------------------
# 📊 MODEL COMPARISON CHART
# ---------------------------
lr_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "CO2": lr_pred,
    "Model": "Linear Regression"
})

lstm_df = pd.DataFrame({
    "Year": lstm_years,
    "CO2": lstm_pred,
    "Model": "LSTM (Deep Learning)"
})

combined = pd.concat([lr_df, lstm_df])

st.subheader("📊 Model Comparison Chart")

fig2 = px.line(
    combined,
    x="Year",
    y="CO2",
    color="Model",
    markers=True,
    title="Linear Regression vs LSTM Forecast Comparison"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# 📉 EVALUATION METRICS
# ---------------------------
st.subheader("📉 Model Evaluation Metrics")

min_len = min(len(lr_pred), len(lstm_pred))

lr_trim = lr_pred[:min_len]
lstm_trim = lstm_pred[:min_len]

mae = mean_absolute_error(lr_trim, lstm_trim)
rmse = np.sqrt(mean_squared_error(lr_trim, lstm_trim))

st.write(f"""
- 📊 MAE (LR vs LSTM): **{mae:.2f}**
- 📊 RMSE (LR vs LSTM): **{rmse:.2f}**
""")

# ---------------------------
# 🌍 GLOBAL MAP
# ---------------------------
st.header("🌍 Global Emissions Map")

latest = df[df['year'] == df['year'].max()]

fig3 = px.choropleth(
    latest,
    locations="country",
    locationmode="country names",
    color="co2",
    color_continuous_scale="Reds",
    title="Global CO₂ Emissions"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# ⚙️ SCENARIO SIMULATOR
# ---------------------------
st.header("⚙️ Climate Scenario Simulator")

st.write("This simulates impact of emission reduction policies on future CO₂ trends.")

reduction = st.slider("Emission Reduction (%)", 0, 100, 20)

scenario = lr_df.copy()
scenario["Adjusted CO2"] = scenario["CO2"] * (1 - reduction / 100)

fig4 = px.line(
    scenario,
    x="Year",
    y="Adjusted CO2",
    title="Policy Impact Simulation"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# 🧠 AI INSIGHTS
# ---------------------------
st.header("🧠 AI Insights")

trend = "increasing" if lr_pred[-1] > lr_pred[0] else "decreasing"
volatility = np.std(c_df["co2"])

st.write(f"""
- 📈 Trend Direction: **{trend}**
- 📊 Emission Volatility: **{volatility:.2f}**
- 🤖 Models: Linear Regression + LSTM Deep Learning
- 🌍 Insight: Emissions show non-linear climate-economic behavior
- ⚠️ Policy Impact: Highly sensitive to reduction scenarios
""")

st.caption("Research Prototype: ML vs Deep Learning comparison for climate forecasting systems.")
