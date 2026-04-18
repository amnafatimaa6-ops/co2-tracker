import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CO₂ Climate Tracker", layout="wide")

st.title("🌍 CO₂ Climate Tracker + Forecast System")
st.write("AI-powered climate intelligence dashboard")

# -------------------------
# LOAD DATA (CLEAN + SAFE)
# -------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)

    df = df[['country', 'year', 'co2']].dropna()
    return df

df = load_data()

# -------------------------
# SIDEBAR
# -------------------------
country = st.sidebar.selectbox("Select Country", df['country'].unique())

country_df = df[df['country'] == country]

# -------------------------
# HISTORICAL TREND
# -------------------------
st.header("📊 Historical CO₂ Emissions")

fig = px.line(country_df, x="year", y="co2",
              title=f"{country} CO₂ Emissions Over Time")

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# GLOBAL MAP
# -------------------------
st.header("🗺️ Global CO₂ Map")

latest_year = df['year'].max()
map_df = df[df['year'] == latest_year]

fig2 = px.choropleth(
    map_df,
    locations="country",
    locationmode="country names",
    color="co2",
    color_continuous_scale="Reds",
    title="Global CO₂ Emissions (Latest Year)"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# SIMPLE FORECAST (TREND MODEL)
# -------------------------
st.header("🔮 Future Trend Forecast")

from sklearn.linear_model import LinearRegression
import numpy as np

X = country_df['year'].values.reshape(-1, 1)
y = country_df['co2'].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(2025, 2051).reshape(-1, 1)
preds = model.predict(future_years)

forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted CO2": preds
})

fig3 = px.line(forecast_df, x="Year", y="Predicted CO2",
               title="CO₂ Emission Forecast")

st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# SCENARIO SIMULATOR
# -------------------------
st.header("⚙️ Emission Reduction Simulator")

reduction = st.slider("Reduce emissions (%)", 0, 100, 20)

forecast_df["Adjusted CO2"] = forecast_df["Predicted CO2"] * (1 - reduction / 100)

fig4 = px.line(forecast_df, x="Year", y="Adjusted CO2",
               title="What-if Scenario")

st.plotly_chart(fig4, use_container_width=True)
