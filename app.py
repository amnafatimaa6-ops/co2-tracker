import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_data, get_country_data
from model.forecast import train_forecast, predict_future

st.set_page_config(page_title="CO₂ Climate Tracker", layout="wide")

st.title("🌍 CO₂ Climate Tracker + Forecast System")
st.subheader("AI-powered climate intelligence dashboard")

# Load data
df = load_data()

# Sidebar
country = st.sidebar.selectbox("Select Country", df['country'].unique())

country_df = get_country_data(df, country)

# -----------------------------
# 📈 Historical Trend
# -----------------------------
st.header("📊 Historical CO₂ Emissions")

fig = px.line(country_df, x="year", y="co2", title=f"{country} CO₂ Emissions Over Time")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🤖 Forecast
# -----------------------------
st.header("🔮 Future Emission Forecast")

model = train_forecast(country_df)
years, preds = predict_future(model)

forecast_df = pd.DataFrame({
    "Year": years,
    "Predicted CO2": preds
})

fig2 = px.line(forecast_df, x="Year", y="Predicted CO2",
               title="Future CO₂ Emission Prediction")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 🌍 World Map (GLOBAL VIEW)
# -----------------------------
st.header("🗺️ Global CO₂ Map")

latest_year = df['year'].max()
map_df = df[df['year'] == latest_year]

fig3 = px.choropleth(
    map_df,
    locations="country",
    locationmode="country names",
    color="co2",
    title="Global CO₂ Emissions (Latest Year)",
    color_continuous_scale="Reds"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# ⚙️ Scenario Simulation
# -----------------------------
st.header("⚙️ Emission Scenario Simulator")

reduction = st.slider("Emission Reduction (%)", 0, 100, 20)

adjusted = forecast_df.copy()
adjusted["Adjusted CO2"] = adjusted["Predicted CO2"] * (1 - reduction/100)

fig4 = px.line(adjusted, x="Year", y="Adjusted CO2",
               title="What-if Scenario: Reduced Emissions")
st.plotly_chart(fig4, use_container_width=True)
