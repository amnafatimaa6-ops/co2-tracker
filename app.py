import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
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

fig1 = px.line(c_df, x="year", y="co2", title=f"{country} CO₂ Emissions Over Time")
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------
# 🤖 MODEL COMPARISON
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

fig2 = px.line(
    combined,
    x="Year",
    y="CO2",
    color="Model",
    title="Climate Forecasting: Classical ML vs Deep Learning (LSTM)"
)

st.plotly_chart(fig2, use_container_width=True)

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

fig4 = px.line(scenario, x="Year", y="Adjusted CO2",
               title="Policy Impact Simulation")

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# 🧠 AI INSIGHTS
# ---------------------------
st.header("🧠 AI Insights")

trend = "increasing" if lr_pred[-1] > lr_pred[0] else "decreasing"
volatility = np.std(c_df["co2"])
model_gap = np.mean(np.abs(lr_pred[:len(lstm_pred)] - lstm_pred[:len(lr_pred)]))

st.write(f"""
- 📈 Trend Direction: **{trend}**
- 📊 Emission Volatility: **{volatility:.2f}**
- ⚖️ Model Divergence (LR vs LSTM): **{model_gap:.2f}**
- 🌍 Interpretation: Emissions show non-linear climate-economic behavior
- ⚠️ Insight: Policy changes have delayed but significant long-term effects
""")

st.caption("Research Prototype: ML vs Deep Learning comparison for climate forecasting systems.")
