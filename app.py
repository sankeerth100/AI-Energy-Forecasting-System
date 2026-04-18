import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Energy AI Dashboard", layout="wide")

# -------------------------
# DARK THEME (UI STYLE)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
.kpi-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<h1 style='text-align: center;'>⚡ AI Energy Analytics Dashboard</h1>
<p style='text-align: center; color: gray;'>
Real-time energy consumption insights & forecasting
</p>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
if not os.path.exists("models/energy_model.pkl"):
    st.error("❌ Model not found. Run main.py first.")
    st.stop()

model = joblib.load("models/energy_model.pkl")

# -------------------------
# FILE UPLOAD
# -------------------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded")
else:
    df = pd.read_csv("data/energy_consumption.csv")

# -------------------------
# VALIDATION
# -------------------------
if "datetime" not in df.columns or "energy" not in df.columns:
    st.error("CSV must contain 'datetime' and 'energy' columns")
    st.stop()

df["datetime"] = pd.to_datetime(df["datetime"])

# -------------------------
# FILTERS
# -------------------------
st.sidebar.header("📅 Filters")

start_date = st.sidebar.date_input("Start Date", df["datetime"].min())
end_date = st.sidebar.date_input("End Date", df["datetime"].max())

df = df[
    (df["datetime"].dt.date >= start_date) &
    (df["datetime"].dt.date <= end_date)
]

# -------------------------
# KPI CARDS
# -------------------------
st.markdown("## 📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Total Energy</h3>
        <h2>{df['energy'].sum():.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Average Energy</h3>
        <h2>{df['energy'].mean():.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Peak Energy</h3>
        <h2>{df['energy'].max():.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek

df["lag_1"] = df["energy"].shift(1)
df["lag_2"] = df["energy"].shift(2)
df["lag_24"] = df["energy"].shift(24)
df["rolling_mean_3"] = df["energy"].rolling(3).mean()

df.dropna(inplace=True)

# -------------------------
# TABS (PRO UI)
# -------------------------
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Predictions", "🔮 Forecast"])

# =========================
# 📈 TAB 1 — DASHBOARD
# =========================
with tab1:

    st.markdown("### 📈 Energy Trends")

    fig1 = px.line(df, x="datetime", y="energy")
    st.plotly_chart(fig1, use_container_width=True)

    colA, colB = st.columns(2)

    with colA:
        hourly = df.groupby("hour")["energy"].mean().reset_index()
        fig2 = px.bar(hourly, x="hour", y="energy", title="Hourly Pattern")
        st.plotly_chart(fig2, use_container_width=True)

    with colB:
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["energy"].mean().reset_index()
        fig3 = px.line(daily, x="date", y="energy", title="Daily Pattern")
        st.plotly_chart(fig3, use_container_width=True)

# =========================
# 🤖 TAB 2 — PREDICTIONS
# =========================
with tab2:

    st.markdown("### 🤖 Model Predictions")

    if os.path.exists("outputs/predictions.csv"):
        pred_df = pd.read_csv("outputs/predictions.csv")

        fig4 = px.line(pred_df, y=["Actual", "Predicted"])
        st.plotly_chart(fig4, use_container_width=True)

        st.download_button(
            "📥 Download Predictions",
            pred_df.to_csv(index=False),
            "predictions.csv"
        )
    else:
        st.warning("Run main.py first to generate predictions")

    st.markdown("### 🔥 Feature Importance")

    features = ["hour","day","month","dayofweek","lag_1","lag_2","lag_24","rolling_mean_3"]
    importances = model.feature_importances_

    fi_df = pd.DataFrame({"Feature": features, "Importance": importances})

    fig5 = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig5, use_container_width=True)

# =========================
# 🔮 TAB 3 — FORECAST
# =========================
with tab3:

    st.markdown("### 🔮 Future Forecast")

    if st.button("Generate 24-Hour Forecast"):

        last = df.iloc[-1]
        preds = []
        temp = last.copy()

        for i in range(24):
            input_data = [[
                temp["hour"],
                temp["day"],
                temp["month"],
                temp["dayofweek"],
                temp["energy"],
                temp["energy"],
                temp["energy"],
                temp["energy"]
            ]]
            pred = model.predict(input_data)[0]
            preds.append(pred)

        forecast_df = pd.DataFrame({"Step": range(24), "Energy": preds})

        fig6 = px.line(forecast_df, x="Step", y="Energy")
        st.plotly_chart(fig6, use_container_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<hr>
<p style='text-align: center; color: gray;'>
Built with ❤️ using Machine Learning & Streamlit
</p>
""", unsafe_allow_html=True)