import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

st.set_page_config(page_title="CO₂ Climate Intelligence System", layout="wide")

st.title("🌍 CO₂ Climate Intelligence System")
st.write("Scholarship-grade AI climate analytics platform")

# ----------------------------
# DATA (LIVE = PROFESSIONAL)
# ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)
    df = df[['country', 'year', 'co2']].dropna()
    return df

df = load_data()

# ----------------------------
# COUNTRY SELECT
# ----------------------------
country = st.sidebar.selectbox("Select Country", df['country'].unique())
c_df = df[df['country'] == country]

# ----------------------------
# 📊 TREND ANALYSIS
# ----------------------------
st.header("📊 Historical Emissions")

fig = px.line(c_df, x="year", y="co2", title=f"{country} CO₂ Trend")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 🤖 FORECAST MODEL
# ----------------------------
st.header("🔮 AI Forecast Model")

X = c_df['year'].values.reshape(-1, 1)
y = c_df['co2'].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(2025, 2051).reshape(-1, 1)
preds = model.predict(future_years)

forecast = pd.DataFrame({
    "Year": future_years.flatten(),
    "CO2": preds
})

fig2 = px.line(forecast, x="Year", y="CO2", title="Emission Forecast (AI)")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 🌍 GLOBAL CLUSTERING
# ----------------------------
st.header("🌍 Country Emission Clusters")

latest = df[df['year'] == df['year'].max()]
cluster_data = latest[['co2']].dropna()

kmeans = KMeans(n_clusters=4, n_init=10)
latest['cluster'] = kmeans.fit_predict(cluster_data)

fig3 = px.scatter(latest, x="country", y="co2", color="cluster",
                  title="Emission Behaviour Clusters")
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# ⚙️ SCENARIO SIMULATION
# ----------------------------
st.header("⚙️ Climate Scenario Simulator")

reduction = st.slider("Emission Reduction (%)", 0, 100, 20)

forecast["Adjusted CO2"] = forecast["CO2"] * (1 - reduction/100)

fig4 = px.line(forecast, x="Year", y="Adjusted CO2",
               title="Net-Zero Scenario Simulation")

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🧠 INSIGHTS PANEL
# ----------------------------
st.header("🧠 AI Insights")

trend = "increasing" if preds[-1] > preds[0] else "decreasing"

st.write(f"""
- 📈 Current trend: **{trend}**
- 🌍 Latest emission snapshot included
- 🤖 Model: Linear Regression forecasting
- ⚠️ Scenario analysis shows impact of policy changes
""")
