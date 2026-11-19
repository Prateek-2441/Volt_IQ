import joblib
import numpy as np
import streamlit as st


class _SingleClassSVMFallback:
    def __init__(self, cls_idx, n_classes):
        self.cls_idx = cls_idx
        self.n_classes = n_classes
        self.classes_ = np.array([cls_idx])

    def predict(self, X):
        return np.full(len(X), self.cls_idx, dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), self.n_classes), dtype=float)
        probs[:, self.cls_idx] = 1.0
        return probs


# Load trained models
@st.cache_resource
def load_models():
    return joblib.load("models/ev_battery_models.joblib")


models = load_models()
def load_models():
    return joblib.load("models/ev_battery_models.joblib")

models = load_models()

# Custom design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #080d12;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #101922;
}

h1, h2, h3 {
    color: #b7f34a !important;
}

div[data-testid="stMetric"] {
    background: #14202a;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #263b42;
}

.stButton button {
    background: #b7f34a;
    color: #080d12;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚡ VOLT//IQ")
st.sidebar.caption("EV BATTERY INTELLIGENCE")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "About the Model"]
)

st.markdown("# VOLT//IQ")
st.markdown("### Intelligent EV Battery Analytics")
st.caption("Machine learning powered battery intelligence system.")

if page == "Dashboard":

    st.markdown("---")
    st.subheader("Battery Parameters")

    col1, col2 = st.columns(2)

    with col1:
        voltage = st.number_input(
            "Voltage (V)", 2.5, 5.0, 4.1, step=0.01
        )
        current = st.number_input(
            "Current (A)", -100.0, 100.0, 20.0
        )
        temperature = st.number_input(
            "Temperature (°C)", -20.0, 100.0, 28.0
        )

    with col2:
        capacity = st.number_input(
            "Capacity (Ah)", 1.0, 100.0, 48.5
        )
        cycle = st.number_input(
            "Cycle Number", 0, 5000, 50
        )
        max_cycles = st.number_input(
            "Maximum Cycles", 1, 10000, 800
        )

    rated_capacity = st.number_input(
        "Rated Capacity (Ah)", 1.0, 100.0, 50.0
    )

    if st.button("⚡ ANALYZE BATTERY"):

        if cycle > max_cycles:
            st.error("Cycle number cannot exceed maximum cycles.")

        else:
            # Feature engineering
            power = voltage * current
            cycle_norm = cycle / max_cycles
            capacity_ratio = capacity / rated_capacity

            rough_soc = np.clip(
                (voltage - 3.0) / 1.2, 0, 1
            ) * 100

            voltage_soc = voltage * (rough_soc / 100)
            abs_current = abs(current)
            temp_deviation = abs(temperature - 25)

            charging = int(current > 1)
            discharging = int(current < -1)
            idle = int(abs(current) <= 1)

            # Regression features
            reg_features = np.array([[
                voltage, current, temperature, capacity,
                cycle, power, cycle_norm, capacity_ratio,
                voltage_soc, abs_current, temp_deviation,
                charging, discharging, idle
            ]])

            # SOH and classification features
            soh_features = np.array([[
                voltage, current, temperature, cycle,
                power, cycle_norm, voltage_soc, abs_current,
                temp_deviation, charging, discharging, idle
            ]])

            try:
                soc = float(models["soc_model"].predict(
                    reg_features
                )[0])

                soh = float(models["soh_model"].predict(
                    soh_features
                )[0])

                scaled = models["condition_scaler"].transform(
                    soh_features
                )

                condition_code = models["condition_model"].predict(
                    scaled
                )[0]

                condition = models["condition_encoder"].inverse_transform(
                    [condition_code]
                )[0]

                soc = np.clip(soc, 0, 100)
                soh = np.clip(soh, 0, 100)

                st.session_state["result"] = {
                    "soc": soc,
                    "soh": soh,
                    "condition": str(condition)
                }

            except Exception as e:
                st.error(f"Prediction error: {e}")

    if "result" in st.session_state:

        result = st.session_state["result"]

        st.markdown("---")
        st.subheader("Battery Intelligence Results")

        a, b, c = st.columns(3)

        a.metric("STATE OF CHARGE", f"{result['soc']:.2f}%")
        b.metric("STATE OF HEALTH", f"{result['soh']:.2f}%")
        c.metric("BATTERY CONDITION", result["condition"])

        st.markdown("### Battery Analytics")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["SOC", "SOH"],
            y=[result["soc"], result["soh"]],
            marker_color=["#b7f34a", "#54d9e8"]
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080d12",
            plot_bgcolor="#101922",
            yaxis=dict(range=[0, 100], title="Percentage"),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("Battery analysis completed.")

        st.caption(
            "Educational ML estimates only. "
            "Not a certified battery safety system."
        )

elif page == "About the Model":

    st.subheader("Our Machine Learning Engine")

    st.markdown("""
    **SOC Prediction**
    
    Random Forest Regressor

    **SOH Prediction**

    Random Forest Regressor

    **Battery Condition Classification**

    Support Vector Machine with RBF Kernel

    **Preprocessing**

    StandardScaler and Label Encoder.

    The underlying notebook uses a synthetic demonstration
    dataset. Predictions are not experimentally validated
    against real EV battery measurements.
    """)

st.markdown("---")
st.caption("VOLT//IQ | EV Battery Intelligence System")
