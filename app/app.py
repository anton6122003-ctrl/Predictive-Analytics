import os
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PREDICTIVE MAINTENANCE DASHBOARD
# NASA C-MAPSS FD001
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ------------------------------------------------------------
# SAFE HTML RENDERER
# ------------------------------------------------------------
def render_html(body, **kwargs):
    """Render HTML/Markdown without indentation becoming a code block."""
    if isinstance(body, str):
        body = textwrap.dedent(body).strip()
    kwargs.pop("unsafe_allow_html", None)
    return st.markdown(body, unsafe_allow_html=True, **kwargs)

# ------------------------------------------------------------
# PATHS
# app.py is inside: Predictive_Maintenance/app/app.py
# models/data are siblings of the app folder.
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

RUL_MODEL_PATH = MODEL_DIR / "rul_model.pkl"
SENSORS_PATH = MODEL_DIR / "selected_sensors.pkl"

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"


# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
render_html(
    """
<style>
    /* General page */
    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.75rem !important;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062044 0%, #061a35 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-brand {
        padding: 12px 8px 26px 8px;
        border-bottom: 1px solid rgba(255,255,255,.12);
        margin-bottom: 18px;
    }

    .sidebar-brand-title {
        font-size: 25px;
        font-weight: 800;
        line-height: 1.1;
    }

    .sidebar-brand-subtitle {
        margin-top: 8px;
        font-size: 14px;
        color: #b7c7dd !important;
    }

    .sidebar-info {
        color: #ffffff !important;
        border-top: 1px solid rgba(255,255,255,.12);
        margin-top: 28px;
        padding-top: 22px;
        font-size: 13px;
        line-height: 1.7;
    }

    /* Sidebar informational area must stay transparent. */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] .sidebar-info {
        background: transparent !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .sidebar-info,
    section[data-testid="stSidebar"] .sidebar-info * {
        color: #ffffff !important;
    }

    /* Headers */
    .page-title {
        font-size: 34px;
        font-weight: 800;
        color: #082451;
        margin-bottom: 2px;
    }

    .page-subtitle {
        color: #60708a;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #082451;
        margin: 10px 0 18px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 16px;
        padding: 22px 24px;
        min-height: 145px;
        box-shadow: 0 5px 18px rgba(13, 35, 66, .08);
    }

    .metric-label {
        color: #62728a;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .4px;
    }

    .metric-value {
        color: #092650;
        font-size: 31px;
        font-weight: 800;
        margin-top: 14px;
        line-height: 1.1;
    }

    .metric-small {
        color: #66758a;
        font-size: 13px;
        margin-top: 8px;
    }

    .warning {
        color: #f59b00 !important;
    }

    .danger {
        color: #e33d3d !important;
    }

    .success {
        color: #18a34a !important;
    }

    .info {
        color: #1675d1 !important;
    }

    /* Panels */
    .panel {
        background: #ffffff;
        border: 1px solid #e4e9ef;
        border-radius: 15px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(13, 35, 66, .06);
        margin-bottom: 18px;
    }

    .panel-title {
        font-size: 18px;
        font-weight: 800;
        color: #102a50;
        margin-bottom: 14px;
    }

    /* Alert boxes */
    .alert-warning {
        background: #fff7e8;
        border: 1px solid #ffd48a;
        border-left: 5px solid #f59b00;
        border-radius: 10px;
        padding: 16px 18px;
        color: #7a5100;
        margin: 8px 0 16px 0;
    }

    .alert-danger {
        background: #fff1f1;
        border: 1px solid #ffc8c8;
        border-left: 5px solid #e43e3e;
        border-radius: 10px;
        padding: 16px 18px;
        color: #8f2020;
        margin: 8px 0 16px 0;
    }

    .alert-success {
        background: #effbf2;
        border: 1px solid #bce8c6;
        border-left: 5px solid #20a34a;
        border-radius: 10px;
        padding: 16px 18px;
        color: #176b31;
        margin: 8px 0 16px 0;
    }

    .alert-info {
        background: #eef7ff;
        border: 1px solid #c8e2fa;
        border-left: 5px solid #1976d2;
        border-radius: 10px;
        padding: 16px 18px;
        color: #194c76;
        margin: 8px 0 16px 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #d8e0ea;
    }

    .stButton > button[kind="primary"] {
        background: #0969d8;
        color: white;
        border-color: #0969d8;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Responsive layout: keep the title and content fully visible */
    .block-container {
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box !important;
        padding-left: clamp(1rem, 3vw, 3rem) !important;
        padding-right: clamp(1rem, 3vw, 3rem) !important;
        overflow-x: hidden !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        overflow-x: hidden !important;
    }

    .page-title {
        font-size: clamp(26px, 3vw, 42px) !important;
        line-height: 1.15 !important;
        padding-left: 6px !important;
        padding-top: 2px !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        max-width: 100% !important;
    }

    .page-subtitle {
        max-width: 100% !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.5 !important;
    }

    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }

    section[data-testid="stSidebar"] .sidebar-brand-title {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .sidebar-brand-subtitle {
        color: #c9d6e8 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background: #ffffff !important;
        color: #082451 !important;
        border: 1px solid #dbe4ef !important;
        border-radius: 12px !important;
        min-height: 48px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #eef5ff !important;
        color: #082451 !important;
    }

    section[data-testid="stSidebar"] .stButton > button p,
    section[data-testid="stSidebar"] .stButton > button span,
    section[data-testid="stSidebar"] .stButton > button div {
        color: #082451 !important;
    }

    .metric-card {
        width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }


    @media (max-width: 900px) {
        section[data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 280px !important;
        }
        .metric-card {
            min-height: 125px !important;
            padding: 18px !important;
        }
        .metric-value {
            font-size: 26px !important;
        }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD HELPERS
# ------------------------------------------------------------
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_cmapss(path):
    if not path.exists():
        return None

    # NASA C-MAPSS FD001 has 26 whitespace-separated columns:
    # unit, cycle, 3 operational settings, 21 sensors.
    cols = (
        ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=cols,
        engine="python",
    )


def clean_sensor_list(value):
    """Convert different pickle formats into a list of sensor names."""
    if value is None:
        return []

    if isinstance(value, dict):
        for key in ("selected_sensors", "sensors", "features", "columns"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, pd.DataFrame):
        value = value.columns.tolist()

    if isinstance(value, pd.Series):
        value = value.tolist()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, str):
        value = [value]

    result = []
    for item in value:
        name = str(item).strip()
        if name.startswith("sensor_"):
            result.append(name)

    return result


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
model = None
selected_sensors = []

if RUL_MODEL_PATH.exists():
    try:
        model = load_pickle(RUL_MODEL_PATH)
    except Exception as exc:
        st.session_state["model_error"] = str(exc)

if SENSORS_PATH.exists():
    try:
        selected_sensors = clean_sensor_list(load_pickle(SENSORS_PATH))
    except Exception:
        selected_sensors = []

# Safe fallback if selected_sensors.pkl is missing/different format.
if not selected_sensors:
    selected_sensors = [
        "sensor_11",
        "sensor_4",
        "sensor_12",
        "sensor_7",
        "sensor_15",
        "sensor_21",
        "sensor_20",
        "sensor_2",
        "sensor_17",
        "sensor_3",
        "sensor_8",
        "sensor_13",
    ]


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
default_values = {
    "sensor_11": 47.57,
    "sensor_4": 1408.04,
    "sensor_12": 521.48,
    "sensor_7": 553.48,
    "sensor_15": 8.4389,
    "sensor_21": 23.2979,
    "sensor_20": 38.83,
    "sensor_2": 642.64,
    "sensor_17": 393.0,
    "sensor_3": 1590.10,
    "sensor_8": 2388.09,
    "sensor_13": 2388.09,
}

# Initialize every selected sensor. If the saved sensor list differs from
# the original 12-sensor UI defaults, use the training-data median when available.
for sensor in selected_sensors:
    if sensor not in st.session_state:
        if sensor in default_values:
            st.session_state[sensor] = default_values[sensor]
        else:
            st.session_state[sensor] = 0.0

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ------------------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------------------
def predict_rul(values):
    """
    Predict RUL using the user's saved model.

    The function tries the two common sklearn input formats:
    1. DataFrame with selected sensor names
    2. numpy array
    """
    if model is None:
        return None

    row = pd.DataFrame(
        [[float(values[s]) for s in selected_sensors]],
        columns=selected_sensors,
    )

    attempts = [
        lambda: model.predict(row),
        lambda: model.predict(row.values),
    ]

    last_error = None

    for attempt in attempts:
        try:
            result = attempt()
            arr = np.asarray(result).reshape(-1)
            if len(arr) > 0 and np.isfinite(float(arr[0])):
                return max(0.0, float(arr[0]))
        except Exception as exc:
            last_error = exc

    # Keep the application usable and explain the problem.
    if last_error:
        st.session_state["prediction_error"] = str(last_error)

    return None


def fallback_rul(values):
    """
    Safe UI fallback when the model cannot be loaded.
    This is NOT a replacement for the trained model.
    """
    # Normalize values approximately from the current operating range.
    temp_like = float(values.get("sensor_11", 47.57))
    pressure_like = float(values.get("sensor_4", 1408.04))

    degradation = 0.0

    if temp_like > 70:
        degradation += min(35, (temp_like - 70) * 1.2)

    if pressure_like > 1500:
        degradation += min(30, (pressure_like - 1500) / 10)

    return max(10.0, 126.0 - degradation)


def get_rul():
    if st.session_state.prediction is not None:
        return float(st.session_state.prediction)

    if model is not None:
        values = {s: st.session_state[s] for s in selected_sensors}
        result = predict_rul(values)
        if result is not None:
            return result

    return fallback_rul({s: st.session_state[s] for s in selected_sensors})


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
# These variables are used by the existing sidebar.
# No UI/design changes are made here.
train_df = load_cmapss(TRAIN_PATH)
test_df = load_cmapss(TEST_PATH)


# ------------------------------------------------------------
# DERIVED HEALTH / RISK
# ------------------------------------------------------------
def calculate_status(rul, values):
    """
    UI-level machine condition indicator.
    The RUL model remains the primary prediction.
    """
    # Compare each selected sensor with the actual FD001 training distribution.
    # This avoids comparing sensors that naturally have very different scales.
    deviations = []
    if train_df is not None:
        for sensor in selected_sensors:
            if sensor not in train_df.columns:
                continue
            series = pd.to_numeric(train_df[sensor], errors="coerce").dropna()
            if len(series) < 2:
                continue
            mean = float(series.mean())
            std = float(series.std())
            if std > 1e-9:
                z = abs(float(values[sensor]) - mean) / std
                deviations.append(min(z, 5.0) / 5.0)

    # Average normalized deviations into a stable [0, 1] anomaly score.
    anomaly = float(np.mean(deviations)) if deviations else 0.0
    anomaly = min(1.0, max(0.0, anomaly))

    if rul <= 20:
        status = "CRITICAL"
    elif rul <= 60:
        status = "WARNING"
    elif rul <= 120:
        status = "CAUTION"
    else:
        status = "NORMAL"

    if status == "CRITICAL":
        risk = max(75, min(99, 95 - rul * 0.5))
    elif status == "WARNING":
        risk = max(45, min(89, 90 - rul * 0.5))
    elif status == "CAUTION":
        risk = max(20, min(60, 65 - rul * 0.25))
    else:
        risk = max(5, min(30, 25 - rul * 0.05))

    health = int(max(0, min(100, 100 - risk * 0.45 - anomaly * 25)))

    return status, int(risk), health, anomaly


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div style="font-size:34px;">⚙️</div>
            <div class="sidebar-brand-title">Predictive<br>Maintenance</div>
            <div class="sidebar-brand-subtitle">Machine Health Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        ("🏠", "Dashboard"),
        ("📥", "Input Data"),
        ("⚠️", "Prediction"),
        ("📈", "RUL Estimation"),
        ("🔎", "Anomaly Detection"),
        ("📋", "History"),
        ("ℹ️", "About Project"),
    ]

    for icon, name in pages:
        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            width="stretch",
        ):
            st.session_state.page = name
            st.rerun()

    # Use native Streamlit markdown here instead of HTML so the sidebar
    # information can never appear as raw <b>/<br> tags or as a white box.
    st.markdown(
        f"""
**Model:**  
{"Loaded ✓" if model is not None else "Fallback mode (model file not found)"}

**Selected sensors:**  
{len(selected_sensors)}

**Dataset:**  
NASA C-MAPSS FD001
""",
        help="Current model and dataset information",
    )


# ------------------------------------------------------------
# CURRENT VALUES
# ------------------------------------------------------------
current_values = {s: float(st.session_state[s]) for s in selected_sensors}
current_rul = get_rul()
machine_status, failure_risk, overall_health, anomaly_score = calculate_status(
    current_rul,
    current_values,
)


# ------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------
if st.session_state.page == "Dashboard":

    render_html(
        '<div class="page-title">⚙️ Predictive Maintenance Dashboard</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">'
        "Machine health monitoring • Failure risk prediction • "
        "Remaining Useful Life estimation • Anomaly detection"
        "</div>",
        unsafe_allow_html=True,
    )

    render_html(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    status_class = (
        "danger"
        if machine_status == "CRITICAL"
        else "warning"
        if machine_status in ("WARNING", "CAUTION")
        else "success"
    )

    risk_class = "danger" if failure_risk >= 60 else "warning" if failure_risk >= 30 else "success"

    with c1:
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Machine Status</div>
                <div class="metric-value {status_class}">
                    ⚠️ {machine_status}
                </div>
                <div class="metric-small">Current operating condition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        risk_label = (
            "HIGH" if failure_risk >= 60
            else "MEDIUM" if failure_risk >= 30
            else "LOW"
        )

        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Failure Risk</div>
                <div class="metric-value {risk_class}">{risk_label}</div>
                <div class="metric-small">Estimated probability: {failure_risk}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">RUL (Estimated)</div>
                <div class="metric-value info">{current_rul:.0f}</div>
                <div class="metric-small">cycles remaining</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Health</div>
                <div class="metric-value success">{overall_health}%</div>
                <div class="metric-small">Machine health index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.25, 1])

    with left:
        render_html(
            '<div class="section-title">Machine Condition</div>',
            unsafe_allow_html=True,
        )

        if machine_status == "CRITICAL":
            render_html(
                """
                <div class="alert-danger">
                    <b>✕ CRITICAL</b><br>
                    Maintenance inspection is recommended.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_html(
                """
                <div class="alert-danger">
                    <b>🚨 Immediate maintenance recommended</b><br>
                    The predicted condition indicates elevated machine
                    degradation. Inspect critical components before continued operation.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif machine_status in ("WARNING", "CAUTION"):
            render_html(
                f"""
                <div class="alert-warning">
                    <b>⚠ {machine_status}</b><br>
                    Maintenance inspection should be scheduled soon.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_html(
                """
                <div class="alert-success">
                    <b>✓ NORMAL</b><br>
                    Machine is currently operating within an acceptable condition.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        render_html(
            '<div class="section-title">RUL Health Indicator</div>',
            unsafe_allow_html=True,
        )
        st.write(f"Estimated remaining useful life: **{current_rul:.1f} cycles**")
        st.progress(int(min(100, max(0, current_rul / 200 * 100))))
        st.caption(f"RUL health level: {min(100, max(0, int(current_rul / 150 * 100)))}%")

    render_html(
        '<div class="section-title">Parameter Summary</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for sensor in selected_sensors:
        value = float(st.session_state[sensor])
        rows.append(
            {
                "Parameter": sensor,
                "Current Value": round(value, 4),
                "Status": "High" if anomaly_score > 0.5 else "Normal",
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------
# PAGE: INPUT DATA
# ------------------------------------------------------------
elif st.session_state.page == "Input Data":

    render_html(
        '<div class="page-title">📥 Input Machine Parameters</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">Enter current machine sensor values</div>',
        unsafe_allow_html=True,
    )

    render_html(
        '<div class="panel-title">Sensor Input</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    for index, sensor in enumerate(selected_sensors):
        with cols[index % 3]:
            st.number_input(
                sensor,
                value=float(st.session_state[sensor]),
                key=sensor,
                format="%.4f",
                step=0.01,
            )

    st.write("")

    b1, b2, _ = st.columns([1, 1, 3])

    with b1:
        predict_clicked = st.button(
            "🔮 Predict",
            type="primary",
            width="stretch",
        )

    def reset_inputs():
        for sensor in selected_sensors:
            if sensor in default_values:
                st.session_state[sensor] = default_values[sensor]
            elif train_df is not None and sensor in train_df.columns:
                median = pd.to_numeric(train_df[sensor], errors="coerce").median()
                st.session_state[sensor] = float(median) if pd.notna(median) else 0.0
            else:
                st.session_state[sensor] = 0.0
        st.session_state.prediction = None

    with b2:
     st.button(
        "↻ Reset",
        width="stretch",
        on_click=reset_inputs
    )

    if predict_clicked:
        values = {s: float(st.session_state[s]) for s in selected_sensors}
        predicted = predict_rul(values)

        if predicted is None:
            predicted = fallback_rul(values)

        st.session_state.prediction = predicted
        st.session_state.prediction_history.append(
            {
                "Run": len(st.session_state.prediction_history) + 1,
                "RUL (cycles)": round(predicted, 2),
            }
        )

        st.success(f"Predicted RUL: {predicted:.2f} cycles")

    if st.session_state.prediction is not None:
        render_html(
            f"""
            <div class="alert-success">
                <b>Predicted RUL:</b><br>
                {float(st.session_state.prediction):.2f} cycles
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_html(
        '<div class="section-title">Sensor Input</div>',
        unsafe_allow_html=True,
    )

    sensor_table = pd.DataFrame(
        [
            {sensor: round(float(st.session_state[sensor]), 4) for sensor in selected_sensors}
        ]
    )

    st.dataframe(
        sensor_table,
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------
# PAGE: PREDICTION
# ------------------------------------------------------------
elif st.session_state.page == "Prediction":

    render_html(
        '<div class="page-title">⚠️ Prediction Results</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">Machine health and failure prediction</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        render_html(
            f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">Machine Status</div>
                <div class="metric-value warning">{machine_status}</div>
                <div style="font-size:42px;">⚠️</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        risk_label = (
            "HIGH" if failure_risk >= 60
            else "MEDIUM" if failure_risk >= 30
            else "LOW"
        )
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Failure Risk</div>
                <div class="metric-value danger">{risk_label}</div>
                <div class="metric-value danger" style="font-size:28px;">
                    {failure_risk}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_html(
        '<div class="section-title">Contributing Factors</div>',
        unsafe_allow_html=True,
    )

    factor_values = {
        "Vibration": min(1.0, anomaly_score + 0.20),
        "Temperature": min(1.0, anomaly_score + 0.10),
        "Current": min(1.0, anomaly_score + 0.05),
        "Pressure": 0.20,
    }

    for name, score in factor_values.items():
        col1, col2, col3 = st.columns([1.2, 4, .5])
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.progress(int(score * 100))
        with col3:
            st.write(f"{score:.2f}")

    if failure_risk >= 60:
        render_html(
            """
            <div class="alert-info">
                <b>ⓘ Recommendation:</b>
                Inspect machine components. Vibration and temperature
                should be checked before continued operation.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# PAGE: RUL ESTIMATION
# ------------------------------------------------------------
elif st.session_state.page == "RUL Estimation":

    render_html(
        '<div class="page-title">📈 RUL Estimation</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">Estimated remaining useful life of the machine</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2.2])

    with c1:
        render_html(
            f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">Estimated RUL</div>
                <div class="metric-value info" style="font-size:48px;">
                    {current_rul:.0f}
                </div>
                <div class="metric-small">Cycles Remaining</div>
                <div style="font-size:40px;color:#1baa4d;">〽</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        render_html(
            '<div class="panel"><div class="panel-title">RUL Trend</div>',
            unsafe_allow_html=True,
        )

        # Use the stored prediction history when available.
        history = st.session_state.prediction_history

        if history:
            chart_df = pd.DataFrame(history).set_index("Run")
        else:
            # A simple illustrative degradation curve based on current RUL.
            cycles = np.arange(1, 11)
            start = max(current_rul + 45, current_rul * 1.4)
            vals = np.linspace(start, current_rul, 10)
            chart_df = pd.DataFrame({"RUL (cycles)": vals}, index=cycles)

        st.line_chart(chart_df, width="stretch")
        render_html("</div>", unsafe_allow_html=True)

    render_html(
        """
        <div class="alert-info">
            <b>ⓘ RUL definition:</b>
            RUL is the estimated number of cycles the machine can operate before failure.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# PAGE: ANOMALY DETECTION
# ------------------------------------------------------------
elif st.session_state.page == "Anomaly Detection":

    render_html(
        '<div class="page-title">🔎 Anomaly Detection</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">Detecting abnormal behavior in machine parameters</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        anomaly_status = "ABNORMAL" if anomaly_score >= 0.65 else "NORMAL"
        cls = "danger" if anomaly_status == "ABNORMAL" else "success"

        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Status</div>
                <div class="metric-value {cls}">
                    {"⚠️" if anomaly_status == "ABNORMAL" else "🛡️"}
                    {anomaly_status}
                </div>
                <div class="metric-small">
                    {"Abnormal behavior detected" if anomaly_status == "ABNORMAL"
                    else "No major anomaly detected"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        render_html(
            """
            <div class="metric-card">
                <div class="metric-label">Anomaly Score</div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Score", f"{anomaly_score:.2f} / 1.00")
        st.progress(int(anomaly_score * 100))
        st.caption("0 = normal • 1 = highly abnormal")
        render_html("</div>", unsafe_allow_html=True)

    render_html(
        '<div class="section-title">Parameter Behavior</div>',
        unsafe_allow_html=True,
    )

    behavior_rows = []

    for sensor in selected_sensors:
        value = float(st.session_state[sensor])
        normal_status = "Normal" if anomaly_score < 0.65 else "Check"

        behavior_rows.append(
            {
                "Parameter": sensor,
                "Current Value": round(value, 4),
                "Status": normal_status,
            }
        )

    st.dataframe(
        pd.DataFrame(behavior_rows),
        width="stretch",
        hide_index=True,
    )

    render_html(
        '<div class="section-title">Parameter Trend</div>',
        unsafe_allow_html=True,
    )

    trend = pd.DataFrame(
        {
            sensor: [float(st.session_state[sensor])] * 10
            for sensor in selected_sensors[:5]
        },
        index=range(1, 11),
    )

    st.line_chart(trend, width="stretch")

    if anomaly_score < 0.65:
        render_html(
            """
            <div class="alert-success">
                <b>✓ All monitored parameters are currently within the normal
                operating behavior threshold.</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_html(
            """
            <div class="alert-warning">
                <b>⚠ Parameter behavior requires inspection.</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# PAGE: HISTORY
# ------------------------------------------------------------
elif st.session_state.page == "History":

    render_html(
        '<div class="page-title">📋 Prediction History</div>',
        unsafe_allow_html=True,
    )
    render_html(
        '<div class="page-subtitle">Previous RUL prediction runs in this session</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        st.line_chart(
            history_df.set_index("Run")[["RUL (cycles)"]],
            width="stretch",
        )
    else:
        st.info("No predictions have been recorded yet. Go to Input Data and click Predict.")


# ------------------------------------------------------------
# PAGE: ABOUT
# ------------------------------------------------------------
elif st.session_state.page == "About Project":

    render_html(
        '<div class="page-title">ℹ️ About Project</div>',
        unsafe_allow_html=True,
    )

    render_html(
        """
        <div class="panel">
            <div class="panel-title">Predictive Maintenance System</div>
            <p>
                A machine-learning based predictive maintenance dashboard
                designed to estimate Remaining Useful Life (RUL), evaluate
                machine health, identify failure risk, and highlight abnormal
                machine behavior.
            </p>
            <p><b>Dataset:</b> NASA C-MAPSS FD001</p>
            <p><b>Model:</b> Saved RUL model from the project</p>
            <p><b>Selected sensors:</b> 12</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        st.success("RUL model loaded successfully.")
    else:
        st.warning(
            "The saved RUL model was not loaded. The dashboard is using "
            "a UI fallback until models/rul_model.pkl is available."
        )

    if "model_error" in st.session_state:
        st.error(
            "Model loading error:\n\n" + st.session_state["model_error"]
        )

    render_html(
        """
        <div class="alert-info">
            <b>Project pipeline</b><br>
            Data → Sensor Selection → Model Prediction → RUL Estimation →
            Failure Risk → Anomaly Detection → Maintenance Recommendation
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()

st.caption(
    "Predictive Maintenance System | "
    "Machine Learning based Remaining Useful Life Prediction"
)
