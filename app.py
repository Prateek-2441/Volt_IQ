import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="VOLT // IQ | Battery Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Fallback class in case the saved bundle references it
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


@st.cache_resource
def load_models():
    return joblib.load("models/ev_battery_models.joblib")


@st.cache_data
def generate_reference_dataset(random_state=42):
    """Generates the same synthetic EV battery dataset from Section 5 of the notebook."""
    np.random.seed(random_state)
    n_batteries = 10
    n_cycles_max = 800
    rated_capacity = 50.0

    records = []
    for bat_id in range(1, n_batteries + 1):
        n_cycles = np.random.randint(200, n_cycles_max + 1)
        for cycle in range(1, n_cycles + 1):
            fade_rate = np.random.uniform(0.00008, 0.00015)
            soh_base = max(0.60, 1.0 - fade_rate * cycle)
            soh = float(np.clip(soh_base + np.random.normal(0, 0.005), 0.60, 1.02))

            capacity = float(np.clip(soh * rated_capacity + np.random.normal(0, 0.3), 28.0, 52.0))
            soc_true = np.random.uniform(0.05, 0.98)
            soc = float(np.clip(soc_true + np.random.normal(0, 0.01), 0.02, 1.0))

            voltage_ocv = 3.0 + 1.2 * soc - 0.3 * soc**2 + np.random.normal(0, 0.03)
            voltage = float(np.clip(voltage_ocv, 2.8, 4.25))

            current = float(np.random.uniform(-60.0, 60.0) + np.random.normal(0, 2.0))
            temp_base = 25 + 5 * abs(current) / 60.0 + np.random.normal(0, 2.0)
            temperature = float(np.clip(temp_base, 10.0, 55.0))

            soh_pct = round(soh * 100, 2)
            condition = "Healthy" if soh_pct >= 80 else ("Warning" if soh_pct >= 70 else "Degraded")

            records.append({
                "battery_id": bat_id,
                "cycle_number": cycle,
                "voltage_v": round(voltage, 4),
                "current_a": round(current, 4),
                "temperature_c": round(temperature, 4),
                "capacity_ah": round(capacity, 4),
                "soc": round(soc * 100, 2),
                "soh": soh_pct,
                "condition": condition,
            })
    return pd.DataFrame(records)


def apply_dark_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6edf3", size=15)),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=height,
        margin=dict(l=40, r=25, t=50, b=40),
        xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    )
    return fig


models = load_models()
ref_df = generate_reference_dataset(42)

# Custom styling matching your VOLT // IQ theme
st.markdown("""
<style>
    .stApp { background-color: #0b0f14; color: #e6edf3; }
    .kpi-card {
        background-color: #161e29;
        border: 1px solid #263241;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .kpi-label { color: #8b949e; font-size: 0.85rem; margin-bottom: 4px; }
    .kpi-value { color: #ffffff; font-size: 2.1rem; font-weight: 700; }
    .section-header { color: #a3e635; font-size: 1.35rem; font-weight: 700; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ VOLT // IQ Controls")
st.sidebar.caption("Adjust live BMS telemetry parameters")

voltage_v = st.sidebar.slider("Terminal Voltage (V)", 2.80, 4.25, 4.10, 0.01)
current_a = st.sidebar.slider("Current (A) [+Charge / -Discharge]", -60.0, 60.0, 20.0, 0.5)
temperature_c = st.sidebar.slider("Cell Temperature (°C)", 10.0, 55.0, 28.0, 0.5)
capacity_ah = st.sidebar.slider("Measured Capacity (Ah)", 28.0, 52.0, 48.5, 0.1)
cycle_number = st.sidebar.slider("Cycle Number", 1, 800, 50, 1)
max_cycles = st.sidebar.number_input("Rated Max Cycles", min_value=200, max_value=1500, value=800, step=50)
rated_capacity = 50.0

# ── Feature Engineering (matches Section 11 & Section 40 of notebook) ──────────
power_w = voltage_v * current_a
cycle_norm = cycle_number / max(max_cycles, 1)
cap_degrade_ratio = capacity_ah / rated_capacity
soc_rough = float(np.clip((voltage_v - 3.0) / 1.2, 0.0, 1.0)) * 100.0
v_soc_interact = voltage_v * (soc_rough / 100.0)
abs_current = abs(current_a)
temp_deviation = abs(temperature_c - 25.0)
mode_charging = 1 if current_a > 1.0 else 0
mode_discharging = 1 if current_a < -1.0 else 0
mode_idle = 1 if abs(current_a) <= 1.0 else 0
charge_mode_label = "Charging" if mode_charging else ("Discharging" if mode_discharging else "Idle")

reg_features = np.array([[
    voltage_v, current_a, temperature_c, capacity_ah,
    cycle_number, power_w, cycle_norm, cap_degrade_ratio,
    v_soc_interact, abs_current, temp_deviation,
    mode_charging, mode_discharging, mode_idle
]])

soh_features = np.array([[
    voltage_v, current_a, temperature_c,
    cycle_number, power_w, cycle_norm,
    v_soc_interact, abs_current, temp_deviation,
    mode_charging, mode_discharging, mode_idle
]])

# ── Predictions using loaded bundle (supports common dict keys) ────────────────
rfr_soc = models.get("rfr_soc") or models.get("rfr")
rfr_soh = models.get("rfr_soh")
svm_clf = models.get("svm_rbf") or models.get("clf")
scaler = models.get("clf_std_scaler") or models.get("scaler")
le_cond = models.get("le_cond")

pred_soc = float(np.clip(rfr_soc.predict(reg_features)[0], 0, 100)) if rfr_soc is not None else soc_rough
pred_soh = float(np.clip(rfr_soh.predict(soh_features)[0], 0, 100)) if rfr_soh is not None else (cap_degrade_ratio * 100)

if svm_clf is not None and scaler is not None and le_cond is not None:
    cond_enc = svm_clf.predict(scaler.transform(soh_features))[0]
    pred_cond = str(le_cond.inverse_transform([cond_enc])[0])
else:
    pred_cond = "Healthy" if pred_soh >= 80 else ("Warning" if pred_soh >= 70 else "Degraded")

cond_badge = {"Healthy": "🟢 Healthy", "Warning": "🟡 Warning", "Degraded": "🔴 Degraded"}.get(pred_cond, pred_cond)

# ── Top KPI Row ────────────────────────────────────────────────────────────────
st.markdown("# ⚡ VOLT // IQ — EV Battery Intelligence System")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">State of Charge (SOC)</div><div class="kpi-value">{pred_soc:.2f}%</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">State of Health (SOH)</div><div class="kpi-value">{pred_soh:.2f}%</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Battery Condition</div><div class="kpi-value">{cond_badge}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Instantaneous Power</div><div class="kpi-value">{power_w:.1f} W</div></div>', unsafe_allow_html=True)

# ── Main Tabs Covering Notebook Analytics ──────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔋 Live Telemetry & Physics",
    "📈 Fleet EDA & Degradation Curves",
    "🤖 ML Model Benchmarks & Feature Importance",
])

# ==============================================================================
# TAB 1: LIVE TELEMETRY, GAUGES, ENGINEERED FEATURES & OCV CURVE
# ==============================================================================
with tab1:
    st.markdown('<div class="section-header">Battery Analytics — State Gauges & Operating Point</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_gauge = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("State of Charge (SOC)", "State of Health (SOH)")
        )
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=pred_soc,
            number={"suffix": "%", "font": {"color": "#a3e635"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#a3e635"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0, 20], "color": "#3b1d24"},
                    {"range": [20, 80], "color": "#1e293b"},
                    {"range": [80, 100], "color": "#1b3329"},
                ],
            }
        ), row=1, col=1)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=pred_soh,
            number={"suffix": "%", "font": {"color": "#4dd0e1"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4dd0e1"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0, 70], "color": "#3b1d24"},
                    {"range": [70, 80], "color": "#3b311d"},
                    {"range": [80, 100], "color": "#1b3329"},
                ],
            }
        ), row=1, col=2)
        apply_dark_layout(fig_gauge, "SOC & SOH Health Threshold Gauges", height=340)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        # Li-ion OCV-SOC Curve + Live Operating Point
        soc_grid = np.linspace(0.02, 1.0, 100)
        ocv_curve = 3.0 + 1.2 * soc_grid - 0.3 * (soc_grid ** 2)
        sample_pts = ref_df.sample(500, random_state=42)

        fig_ocv = go.Figure()
        fig_ocv.add_trace(go.Scatter(
            x=sample_pts["soc"], y=sample_pts["voltage_v"],
            mode="markers", name="Fleet Telemetry",
            marker=dict(size=5, color="#4dd0e1", opacity=0.35)
        ))
        fig_ocv.add_trace(go.Scatter(
            x=soc_grid * 100, y=ocv_curve,
            mode="lines", name="Theoretical Li-ion OCV Curve",
            line=dict(color="#f59e0b", width=2.5, dash="dash")
        ))
        fig_ocv.add_trace(go.Scatter(
            x=[pred_soc], y=[voltage_v],
            mode="markers+text", name="Current Battery",
            text=["Current"], textposition="top center",
            marker=dict(size=14, color="#a3e635", symbol="diamond", line=dict(color="white", width=1.5))
        ))
        apply_dark_layout(fig_ocv, "Li-ion Voltage vs SOC Operating Point", height=340)
        fig_ocv.update_xaxes(title_text="SOC (%)")
        fig_ocv.update_yaxes(title_text="Terminal Voltage (V)")
        st.plotly_chart(fig_ocv, use_container_width=True)

    st.markdown('<div class="section-header">Engineered Physics Features (Section 11)</div>', unsafe_allow_html=True)
    feat_table = pd.DataFrame([
        {"Feature": "power_w", "Value": f"{power_w:.2f} W", "Formula": "Voltage × Current", "Role": "Instantaneous electrical load"},
        {"Feature": "cycle_norm", "Value": f"{cycle_norm:.3f}", "Formula": "Cycle / Max Cycles", "Role": "Normalized lifecycle aging [0–1]"},
        {"Feature": "cap_degrade_ratio", "Value": f"{cap_degrade_ratio:.3f}", "Formula": "Capacity / 50.0 Ah", "Role": "Direct capacity retention ratio"},
        {"Feature": "v_soc_interact", "Value": f"{v_soc_interact:.3f}", "Formula": "Voltage × (SOC / 100)", "Role": "Combined V–SOC energy state"},
        {"Feature": "temp_deviation", "Value": f"{temp_deviation:.2f} °C", "Formula": "|Temp − 25°C|", "Role": "Thermal stress from nominal"},
        {"Feature": "charge_mode", "Value": charge_mode_label, "Formula": "Current threshold (±1 A)", "Role": "One-hot encoded operating mode"},
    ])
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: FLEET EDA, DEGRADATION CURVES, SMOOTHING & CORRELATIONS
# ==============================================================================
with tab2:
    st.markdown('<div class="section-header">Cycle Aging & Multi-Battery Degradation (Section 8)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_deg = go.Figure()
        for bat_id in range(1, 6):
            sub = ref_df[ref_df["battery_id"] == bat_id].sort_values("cycle_number")
            fig_deg.add_trace(go.Scatter(
                x=sub["cycle_number"], y=sub["soh"],
                mode="lines", name=f"Battery {bat_id}",
                line=dict(width=1.8)
            ))
        fig_deg.add_hline(y=80, line_dash="dash", line_color="#22c55e", annotation_text="Healthy (80%)")
        fig_deg.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Degraded (70%)")
        fig_deg.add_trace(go.Scatter(
            x=[cycle_number], y=[pred_soh],
            mode="markers", name="Current Input",
            marker=dict(size=12, color="#a3e635", symbol="star")
        ))
        apply_dark_layout(fig_deg, "Cycle Number vs State of Health (SOH)", height=390)
        fig_deg.update_xaxes(title_text="Cycle Number")
        fig_deg.update_yaxes(title_text="SOH (%)")
        st.plotly_chart(fig_deg, use_container_width=True)

    with c2:
        sample_eda = ref_df.sample(1500, random_state=42)
        fig_temp = px.scatter(
            sample_eda, x="soc", y="voltage_v", color="temperature_c",
            color_continuous_scale="Plasma", opacity=0.55,
            labels={"soc": "SOC (%)", "voltage_v": "Voltage (V)", "temperature_c": "Temp (°C)"}
        )
        apply_dark_layout(fig_temp, "Voltage vs SOC (Colored by Cell Temperature)", height=390)
        st.plotly_chart(fig_temp, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Section 10: Rolling Mean Smoothing on Battery 1
        bat1 = ref_df[ref_df["battery_id"] == 1].sort_values("cycle_number").copy()
        bat1["capacity_smooth"] = bat1["capacity_ah"].rolling(window=10, min_periods=1).mean()
        fig_smooth = go.Figure()
        fig_smooth.add_trace(go.Scatter(
            x=bat1["cycle_number"], y=bat1["capacity_ah"],
            mode="lines", name="Raw Sensor Capacity",
            line=dict(color="#4dd0e1", width=1), opacity=0.45
        ))
        fig_smooth.add_trace(go.Scatter(
            x=bat1["cycle_number"], y=bat1["capacity_smooth"],
            mode="lines", name="Smoothed (Rolling Window=10)",
            line=dict(color="#a3e635", width=2.5)
        ))
        apply_dark_layout(fig_smooth, "Data Smoothing — Raw vs Moving Average Capacity (Battery 1)", height=380)
        fig_smooth.update_xaxes(title_text="Cycle Number")
        fig_smooth.update_yaxes(title_text="Capacity (Ah)")
        st.plotly_chart(fig_smooth, use_container_width=True)

    with c4:
        # Section 8.8: Feature Correlation Heatmap
        corr_cols = ["voltage_v", "current_a", "temperature_c", "capacity_ah", "cycle_number", "soc", "soh"]
        corr = ref_df[corr_cols].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
        )
        apply_dark_layout(fig_corr, "Feature Correlation Matrix", height=380)
        st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 3: MODEL COMPARISON & FEATURE IMPORTANCE
# ==============================================================================
with tab3:
    st.markdown('<div class="section-header">Feature Importances & Model Evaluation (Sections 24–39)</div>', unsafe_allow_html=True)

    fi_col1, fi_col2 = st.columns(2)
    soc_feat_names = [
        "voltage_v", "current_a", "temperature_c", "capacity_ah",
        "cycle_number", "power_w", "cycle_norm", "cap_degrade_ratio",
        "v_soc_interact", "abs_current", "temp_deviation",
        "mode_charging", "mode_discharging", "mode_idle"
    ]
    soh_feat_names = [
        "voltage_v", "current_a", "temperature_c",
        "cycle_number", "power_w", "cycle_norm",
        "v_soc_interact", "abs_current", "temp_deviation",
        "mode_charging", "mode_discharging", "mode_idle"
    ]

    with fi_col1:
        if rfr_soc is not None and hasattr(rfr_soc, "feature_importances_"):
            imp_soc = pd.Series(rfr_soc.feature_importances_, index=soc_feat_names).sort_values()
            fig_fi_soc = go.Figure(go.Bar(
                x=imp_soc.values, y=imp_soc.index, orientation="h", marker_color="#a3e635"
            ))
            apply_dark_layout(fig_fi_soc, "Random Forest Feature Importance — SOC", height=400)
            fig_fi_soc.update_xaxes(title_text="Gini Importance")
            st.plotly_chart(fig_fi_soc, use_container_width=True)

    with fi_col2:
        if rfr_soh is not None and hasattr(rfr_soh, "feature_importances_"):
            imp_soh = pd.Series(rfr_soh.feature_importances_, index=soh_feat_names).sort_values()
            fig_fi_soh = go.Figure(go.Bar(
                x=imp_soh.values, y=imp_soh.index, orientation="h", marker_color="#4dd0e1"
            ))
            apply_dark_layout(fig_fi_soh, "Random Forest Feature Importance — SOH", height=400)
            fig_fi_soh.update_xaxes(title_text="Gini Importance")
            st.plotly_chart(fig_fi_soh, use_container_width=True)

    # Display saved evaluation tables if stored in models dict, otherwise show architecture summary
    if "soc_compare" in models and "soh_compare" in models:
        m1, m2 = st.columns(2)
        with m1:
            st.subheader("SOC Regression Benchmarks")
            st.dataframe(models["soc_compare"], use_container_width=True)
        with m2:
            st.subheader("SOH Regression Benchmarks")
            st.dataframe(models["soh_compare"], use_container_width=True)
    else:
        st.info(
            "Pipeline Summary: Battery-aware Group Split (80/20 by battery_id) • "
            "SOC & SOH Models: Simple LR, Multiple LR, SVR (RBF), Decision Tree, Random Forest • "
            "Condition Classifiers: Logistic Regression, KNN, SVM (Linear & RBF), Gaussian NB, Decision Tree • "
            "Validation: 5-Fold Cross-Validation & 500-Resample Bootstrapping."
        )
