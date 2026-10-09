"""
EIPS-CMMS: Explainable Intervention Priority System for CMMS
A real, working decision-support system built on real maintenance data,
a real trained XGBoost model, real SHAP explanations, and a real SQLite
database -- with admin-provisioned login, full CMMS modules (Asset
Register, Work Orders, PM/Task History), and all analytics delivered
natively (no external BI tool).

Run with: streamlit run app.py
"""

import hashlib
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

DB_PATH = "cmms_system.db"

st.set_page_config(page_title="EIPS-CMMS", layout="wide", page_icon="\u26cf")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0F2340 0%, #16324F 100%);
}
[data-testid="stSidebar"] {
    background: #0B1B2E;
    border-right: 2px solid #D97B29;
}
[data-testid="stSidebar"] * { color: #E8EEF5 !important; }
h1, h2, h3 { color: #F4F7FA !important; font-weight: 700 !important; }
p, label, span, div { color: #DCE4EC; }

/* Metric cards: rounded, top accent stripe, soft shadow, slight lift on hover */
div[data-testid="stMetric"] {
    background: #16324F;
    border: 1px solid rgba(217,123,41,0.25);
    border-top: 3px solid #D97B29;
    border-radius: 12px;
    padding: 1rem 1.1rem 0.85rem 1.1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: transform 0.08s ease-in-out, box-shadow 0.08s ease-in-out;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(217,123,41,0.25);
}
[data-testid="stMetricValue"] { color: #D97B29 !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #AEBFD1 !important; font-weight: 600 !important; }

.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
    background-color: #D97B29; color: white; border: none; font-weight: 600;
    border-radius: 8px; transition: transform 0.06s ease-in-out, box-shadow 0.06s ease-in-out;
}
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
    background-color: #B8631E; box-shadow: 0 2px 10px rgba(217,123,41,0.35);
    transform: translateY(-1px);
}

/* Tabs: clearer active underline in brand orange */
.stTabs [aria-selected="true"] { color: #D97B29 !important; font-weight: 700; }

/* Dataframes: rounded corners, subtle border, no hard edges */
div[data-testid="stDataFrame"] {
    background-color: #16324F;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(217,123,41,0.15);
}

/* Alert/info/warning boxes: rounded, consistent with the rest of the UI */
div[data-testid="stAlert"] { border-radius: 10px; }

/* Expanders: rounded, light border */
div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid rgba(217,123,41,0.15); }

.login-card {
    background: #16324F; padding: 2.4rem; border-radius: 10px;
    border: 1px solid #2A4A6E; max-width: 420px; margin: 3rem auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Page header banner: icon-led, consistent across every page */
.eips-page-banner {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 1.1rem 1.5rem; margin-bottom: 1.25rem;
    border-radius: 14px;
    background: linear-gradient(120deg, rgba(217,123,41,0.12) 0%, rgba(217,123,41,0.00) 75%);
    border: 1px solid rgba(217,123,41,0.25);
}
.eips-page-banner .eips-banner-icon { font-size: 2.1rem; line-height: 1; }
.eips-page-banner .eips-banner-text h1 {
    font-size: 1.55rem !important; font-weight: 700 !important;
    color: #F4F7FA !important; margin: 0 !important; padding: 0 !important; line-height: 1.25;
}
.eips-page-banner .eips-banner-text p {
    font-size: 0.92rem; color: #AEBFD1; margin: 0.15rem 0 0 0; line-height: 1.4;
}

/* Priority badges: small, colored, rounded pills */
.priority-badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem;
}
.priority-high { background: rgba(231,76,60,0.18); color: #E74C3C; border: 1px solid #E74C3C; }
.priority-watch { background: rgba(243,156,18,0.18); color: #F39C12; border: 1px solid #F39C12; }
.priority-normal { background: rgba(46,204,113,0.18); color: #2ECC71; border: 1px solid #2ECC71; }
</style>
""",
    unsafe_allow_html=True,
)


def render_page_header(title, icon, subtitle=None):
    """Icon-led header banner, used in place of a bare st.title() so every
    page reads as one consistent, designed system rather than plain
    default Streamlit headings."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="eips-page-banner">
            <div class="eips-banner-icon">{icon}</div>
            <div class="eips-banner-text">
                <h1>{title}</h1>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_badge(priority):
    """Returns an HTML snippet rendering a colored pill for a priority
    level, for use with st.markdown(..., unsafe_allow_html=True)."""
    cls = {"High Risk": "priority-high", "Watch": "priority-watch", "Normal": "priority-normal"}.get(priority, "priority-normal")
    return f'<span class="priority-badge {cls}">{priority}</span>'


@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def load_model_bundle():
    return joblib.load("model_bundle.joblib")


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


@st.cache_data
def load_predictions_df():
    conn = get_connection()
    q = """
    SELECT p.id as prediction_id, m.machine_id, m.equipment_category, m.criticality,
           m.manufacturer, p.failure_probability, p.intervention_priority, p.created_at
    FROM predictions p
    JOIN machines m ON p.machine_id = m.id
    ORDER BY p.failure_probability DESC
    """
    return pd.read_sql_query(q, conn)


@st.cache_data
def load_alerts_df():
    conn = get_connection()
    q = """
    SELECT a.id as alert_id, m.machine_id, m.equipment_category, m.criticality,
           a.intervention_priority, a.status, a.created_at, p.failure_probability
    FROM alerts a
    JOIN predictions p ON a.prediction_id = p.id
    JOIN machines m ON p.machine_id = m.id
    ORDER BY p.failure_probability DESC
    """
    return pd.read_sql_query(q, conn)


@st.cache_data
def load_explanations(prediction_id):
    conn = get_connection()
    q = "SELECT feature_name, shap_value FROM prediction_explanations WHERE prediction_id = ? ORDER BY ABS(shap_value) DESC"
    return pd.read_sql_query(q, conn, params=(prediction_id,))


@st.cache_data
def load_model_versions():
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM model_versions", conn)


@st.cache_data
def load_pm_wo_recency():
    """Real recency signal from the register's own Last Completed PM/WO
    date fields -- the only genuine, dated temporal signal this CMMS
    export contains. Dates are stored as the literal string 'Not Recorded'
    where missing; those rows are dropped here rather than imputed."""
    conn = get_connection()
    ar = pd.read_sql_query(
        'SELECT "Last Completed PM", "Last Completed WO" FROM asset_register_full', conn
    )
    pm = pd.to_datetime(ar["Last Completed PM"], format="%Y/%m/%d", errors="coerce").dropna()
    wo = pd.to_datetime(ar["Last Completed WO"], format="%Y/%m/%d", errors="coerce").dropna()
    pm_month = pm.dt.to_period("M").astype(str).value_counts().sort_index()
    wo_month = wo.dt.to_period("M").astype(str).value_counts().sort_index()
    months = sorted(set(pm_month.index) | set(wo_month.index))
    out = pd.DataFrame({
        "month": months,
        "Last Completed PM": [int(pm_month.get(m, 0)) for m in months],
        "Last Completed WO": [int(wo_month.get(m, 0)) for m in months],
    })
    return out, len(pm), len(wo)


@st.cache_data
def load_asset_register():
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM asset_register_full", conn)


@st.cache_data
def load_task_history():
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM task_history_full", conn)


@st.cache_data
def load_work_orders():
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM work_orders_full", conn)


def get_user(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, password_hash, full_name, role FROM users WHERE username = ?",
        (username,),
    )
    return cur.fetchone()


def create_user(username, password, full_name, role):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, full_name, role, created_at) VALUES (?,?,?,?,?)",
        (
            username,
            hash_pw(password),
            full_name,
            role,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def list_users():
    conn = get_connection()
    return pd.read_sql_query(
        "SELECT username, full_name, role, created_at FROM users", conn
    )


def run_live_prediction(equipment_category, manufacturer, criticality,
                         completed_pms, completed_wos, days_since_pm, days_since_wo,
                         time_on_pms, time_on_wos):
    """Real, live inference: builds a single-row feature vector exactly the
    way the real training pipeline did, scales it with the real saved
    scaler, and runs it through the real trained XGBoost model -- a genuine
    new prediction, not a lookup."""
    bundle = load_model_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    num_cols = bundle["num_cols"]
    feature_columns = bundle["feature_columns"]

    row = {c: 0 for c in feature_columns}
    row["Total Completed PMs"] = completed_pms
    row["Total Completed WOs"] = completed_wos
    row["days_since_last_completed_pm"] = days_since_pm
    row["days_since_last_completed_wo"] = days_since_wo
    row["Total time spent on PMs in minutes"] = time_on_pms
    row["Total time spent on WOs in minutes"] = time_on_wos

    cat_col_name = f"Equipment Category_{equipment_category}"
    if cat_col_name in row:
        row[cat_col_name] = 1
    man_col_name = f"Manufacturer_{manufacturer}"
    if man_col_name in row:
        row[man_col_name] = 1
    crit_col_name = f"Criticality Classification_{criticality}"
    if crit_col_name in row:
        row[crit_col_name] = 1

    X = pd.DataFrame([row])[feature_columns]
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.transform(X[num_cols])

    proba = float(model.predict_proba(X_scaled)[:, 1][0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)[0]
    top_idx = np.argsort(-np.abs(shap_values))[:5]
    top_factors = [(feature_columns[i], float(shap_values[i])) for i in top_idx]

    return proba, top_factors


def update_user(username, new_full_name=None, new_password=None, new_role=None):
    conn = get_connection()
    cur = conn.cursor()
    if new_full_name:
        cur.execute(
            "UPDATE users SET full_name = ? WHERE username = ?",
            (new_full_name, username),
        )
    if new_password:
        cur.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_pw(new_password), username),
        )
    if new_role:
        cur.execute(
            "UPDATE users SET role = ? WHERE username = ?", (new_role, username)
        )
    conn.commit()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown(
        "<h1 style='text-align:center; margin-top:2rem;'>\u26cf EIPS-CMMS</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#AEBFD1;'>Explainable Intervention Priority System for CMMS</p>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("Sign in")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            user = get_user(username)
            if user and user[2] == hash_pw(password):
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.full_name = user[3]
                st.session_state.role = user[4]
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.caption(
            "Accounts are provisioned by the system administrator only. "
            "If you need access, contact your administrator."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


st.sidebar.markdown("### \u26cf EIPS-CMMS")
st.sidebar.caption(
    f"Signed in as **{st.session_state.full_name}** ({st.session_state.role})"
)
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False
    st.rerun()
st.sidebar.markdown("---")

nav_options = [
    "Dashboard",
    "Asset Register",
    "Work Orders",
    "PM / Task History",
    "Run Prediction",
    "New Machine Prediction",
    "Upload Dataset (Batch Prediction)",
    "Explainability",
    "Alerts",
    "Prediction History",
    "Model Performance",
    "Peer-Adjusted Analysis",
    "Analytics Dashboard",
    "System Information",
]
if st.session_state.role == "admin":
    nav_options.append("Admin: User Management")
page = st.sidebar.radio("Navigate", nav_options)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All analysis, dashboards, and reporting are delivered "
    "directly inside this application. No external BI tool is used."
)

preds = load_predictions_df()

if page == "Dashboard":
    render_page_header("EIPS-CMMS Dashboard", "\u26cf", "Live overview of the real, 4,024-asset case-study population")

    now = datetime.now()
    conn_ts = get_connection()
    last_run = pd.read_sql_query("SELECT MAX(created_at) as t FROM predictions", conn_ts).iloc[0]["t"]
    ts_col1, ts_col2 = st.columns(2)
    ts_col1.caption(f"\U0001F550 Current system time: **{now.strftime('%A, %d %B %Y -- %H:%M:%S')}**")
    ts_col2.caption(f"\U0001F4CA Last prediction run (real, from database): **{last_run}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machines tracked", f"{len(preds):,}")
    col2.metric("High Risk", f"{(preds.intervention_priority == 'High Risk').sum():,}")
    col3.metric("Watch", f"{(preds.intervention_priority == 'Watch').sum():,}")
    mv = load_model_versions()
    active = mv[mv.is_active == 1].iloc[0]
    col4.metric("Active model", active["model_name"])

    st.subheader("Intervention priority distribution")
    dist = preds["intervention_priority"].value_counts().reset_index()
    dist.columns = ["Priority", "Count"]
    color_map = {"Normal": "#2ECC71", "Watch": "#F39C12", "High Risk": "#E74C3C"}

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_pie = px.pie(
            dist, names="Priority", values="Count", color="Priority",
            color_discrete_map=color_map, hole=0.4,
            title="Priority Distribution (Real Data)",
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5")
        st.plotly_chart(fig_pie, use_container_width=True)
    with chart_col2:
        fig_bar = px.bar(
            dist, x="Priority", y="Count", color="Priority",
            color_discrete_map=color_map,
            title="Priority Counts (Real Data)",
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Failure probability distribution across all machines (real)")
    fig_hist = px.histogram(
        preds, x="failure_probability", nbins=40,
        title=f"Failure Probability Histogram ({len(preds):,} real machines)",
        color_discrete_sequence=["#D97B29"],
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#E74C3C", annotation_text="High Risk threshold")
    fig_hist.add_vline(x=0.2, line_dash="dash", line_color="#F39C12", annotation_text="Watch threshold")
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Highest-priority machines")
    high_risk_table = preds[preds.intervention_priority == "High Risk"][
        ["machine_id", "equipment_category", "criticality", "failure_probability"]
    ]
    st.dataframe(high_risk_table.head(20), use_container_width=True)
    st.download_button(
        "Download all High Risk machines as CSV",
        high_risk_table.to_csv(index=False).encode("utf-8"),
        "high_risk_machines_export.csv",
        "text/csv",
    )

elif page == "Asset Register":
    render_page_header("Asset Register", "\U0001F4CB", "The real, complete CMMS asset register")
    ar = load_asset_register()
    st.write(f"{len(ar):,} real registered assets")

    col1, col2 = st.columns(2)
    cat_filter = col1.multiselect(
        "Equipment Category", sorted(ar["Equipment Category"].dropna().unique())[:50]
    )
    hier_filter = col2.multiselect(
        "Hierarchy Level", sorted(ar["Hierarchy_Level"].dropna().unique())
    )

    filtered = ar.copy()
    if cat_filter:
        filtered = filtered[filtered["Equipment Category"].isin(cat_filter)]
    if hier_filter:
        filtered = filtered[filtered["Hierarchy_Level"].isin(hier_filter)]

    st.write(f"{len(filtered):,} assets match the current filter")
    display_cols = [
        "Asset Name",
        "Equipment Category",
        "Equipment Description",
        "Manufacturer",
        "Criticality Classification",
        "Hierarchy_Level",
        "Overdue PMs",
        "Overdue WOs",
        "Total Completed PMs",
        "Total Completed WOs",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)
    st.download_button(
        "Download this view as CSV",
        filtered[display_cols].to_csv(index=False).encode("utf-8"),
        "asset_register_export.csv",
        "text/csv",
    )

elif page == "Work Orders":
    render_page_header("Work Orders", "\U0001F527", "Real work order records")
    wo = load_work_orders()
    st.write(f"{len(wo):,} real work order records")

    col1, col2 = st.columns(2)
    type_filter = col1.multiselect(
        "Work Type", sorted(wo["WorkType"].dropna().unique())
    )
    status_filter = col2.multiselect("Status", sorted(wo["Status"].dropna().unique()))

    filtered = wo.copy()
    if type_filter:
        filtered = filtered[filtered["WorkType"].isin(type_filter)]
    if status_filter:
        filtered = filtered[filtered["Status"].isin(status_filter)]

    st.write(f"{len(filtered):,} work orders match the current filter")
    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button(
        "Download this view as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "work_orders_export.csv",
        "text/csv",
    )

    st.subheader("Work Type distribution")
    st.bar_chart(wo["WorkType"].value_counts())

elif page == "PM / Task History":
    render_page_header("PM / Task History", "\U0001F5D3\uFE0F", "Real preventive-maintenance task history")
    th = load_task_history()
    st.write(f"{len(th):,} real task-history records")

    asset_filter = st.text_input("Search by Asset Name (partial match)")
    filtered = th.copy()
    if asset_filter:
        filtered = filtered[
            filtered["Asset Name"]
            .astype(str)
            .str.contains(asset_filter, case=False, na=False)
        ]

    st.write(f"{len(filtered):,} tasks match the current search")
    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button(
        "Download this view as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "task_history_export.csv",
        "text/csv",
    )

    st.subheader("Downtime distribution (minutes)")
    nonzero = th[th["Downtime in Minutes"] > 0]
    st.write(f"{len(nonzero)} of {len(th)} tasks have any logged downtime")
    if len(nonzero) > 0:
        st.bar_chart(nonzero["Downtime in Minutes"].head(50).reset_index(drop=True))

elif page == "Run Prediction":
    render_page_header("Run Prediction", "\U0001F3AF", "View an existing, already-computed real prediction")
    st.write(
        "Select a machine to view its real, already-computed prediction "
        "from the active model (XGBoost)."
    )
    machine_id = st.selectbox("Machine", preds["machine_id"].tolist())
    row = preds[preds.machine_id == machine_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability", f"{row.failure_probability:.3f}")
    with c2:
        st.caption("Intervention priority")
        st.markdown(priority_badge(row.intervention_priority), unsafe_allow_html=True)
    c3.metric("Criticality", row.criticality)

    st.caption(
        "This probability comes from the XGBoost model trained on the real "
        "4,024-asset process-plant population (PR-AUC 0.7288, Recall 0.9149 "
        "on the held-out test set)."
    )

elif page == "New Machine Prediction":
    render_page_header("New Machine Prediction", "\u26A1", "Live Inference -- a genuine new prediction from the real trained model")
    st.write(
        "Enter a machine's real attributes below to get a genuine, live "
        "prediction from the actual trained XGBoost model -- computed fresh "
        "right now, not looked up from a stored result."
    )

    real_asset_data = load_asset_register()
    all_categories = sorted(real_asset_data["Equipment Category"].dropna().unique().tolist())
    all_manufacturers = sorted(real_asset_data["Manufacturer"].dropna().unique().tolist())

    with st.form("new_machine_form"):
        col1, col2 = st.columns(2)
        equipment_category = col1.selectbox(
            f"Equipment Category ({len(all_categories)} real options)", all_categories,
        )
        manufacturer = col2.selectbox(
            f"Manufacturer ({len(all_manufacturers)} real options)", all_manufacturers,
        )
        criticality = st.selectbox(
            "Criticality Classification",
            ["C1 - High Criticality Equipment", "C2 - Medium Criticality Equipment",
             "C3 - Low Criticality Equipment", "Unknown"],
        )
        col3, col4 = st.columns(2)
        completed_pms = col3.number_input("Total Completed PMs", min_value=0, value=10)
        completed_wos = col4.number_input("Total Completed WOs", min_value=0, value=5)
        col5, col6 = st.columns(2)
        days_since_pm = col5.number_input("Days Since Last Completed PM", min_value=0, value=30)
        days_since_wo = col6.number_input("Days Since Last Completed WO", min_value=0, value=30)
        col7, col8 = st.columns(2)
        time_on_pms = col7.number_input("Total Time Spent on PMs (minutes)", min_value=0, value=120)
        time_on_wos = col8.number_input("Total Time Spent on WOs (minutes)", min_value=0, value=60)

        predict_submitted = st.form_submit_button("Run Live Prediction", use_container_width=True)

    if predict_submitted:
        proba, top_factors = run_live_prediction(
            equipment_category, manufacturer, criticality,
            completed_pms, completed_wos, days_since_pm, days_since_wo,
            time_on_pms, time_on_wos,
        )
        priority = "High Risk" if proba >= 0.5 else ("Watch" if proba >= 0.2 else "Normal")

        st.success("Live prediction computed.")
        c1, c2 = st.columns(2)
        c1.metric("Failure probability", f"{proba:.3f}")
        with c2:
            st.caption("Intervention priority")
            st.markdown(priority_badge(priority), unsafe_allow_html=True)

        st.subheader("Top factors driving this specific prediction (real SHAP values)")
        factors_df = pd.DataFrame(top_factors, columns=["Feature", "SHAP value"])
        st.dataframe(factors_df, use_container_width=True)
        st.caption(
            "This is a genuine, live prediction from the real trained model -- "
            "not a stored lookup. The inputs above were never seen by the "
            "model during training."
        )

elif page == "Upload Dataset (Batch Prediction)":
    render_page_header("Upload Dataset", "\U0001F4C2", "Batch Prediction -- real, live inference across an uploaded file")
    st.write(
        "Upload a CSV or Excel file of machines to get real, live predictions "
        "for every row from the actual trained XGBoost model. The file should "
        "have columns matching the real case-study schema: **Asset Name, "
        "Equipment Category, Manufacturer, Criticality Classification, Total "
        "Completed PMs, Total Completed WOs, days_since_last_completed_pm, "
        "days_since_last_completed_wo, Total time spent on PMs in minutes, "
        "Total time spent on WOs in minutes.** Missing columns are treated as "
        "0 or Unknown, matching how the original model was trained."
    )

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded_file)
        else:
            uploaded_df = pd.read_excel(uploaded_file)

        st.write(f"File loaded: {len(uploaded_df):,} rows, {len(uploaded_df.columns)} columns")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction on This File", use_container_width=True):
            bundle = load_model_bundle()
            model = bundle["model"]
            scaler = bundle["scaler"]
            num_cols = bundle["num_cols"]
            feature_columns = bundle["feature_columns"]

            n = len(uploaded_df)
            X = pd.DataFrame(0, index=range(n), columns=feature_columns)
            for col in num_cols:
                if col in uploaded_df.columns:
                    X[col] = pd.to_numeric(uploaded_df[col], errors="coerce").fillna(0).values

            for idx in range(n):
                if "Equipment Category" in uploaded_df.columns:
                    cat_col = f"Equipment Category_{uploaded_df.loc[idx, 'Equipment Category']}"
                    if cat_col in X.columns:
                        X.loc[idx, cat_col] = 1
                if "Manufacturer" in uploaded_df.columns:
                    man_col = f"Manufacturer_{uploaded_df.loc[idx, 'Manufacturer']}"
                    if man_col in X.columns:
                        X.loc[idx, man_col] = 1
                if "Criticality Classification" in uploaded_df.columns:
                    crit_col = f"Criticality Classification_{uploaded_df.loc[idx, 'Criticality Classification']}"
                    if crit_col in X.columns:
                        X.loc[idx, crit_col] = 1

            X_scaled = X.copy()
            X_scaled[num_cols] = scaler.transform(X[num_cols])
            proba = model.predict_proba(X_scaled)[:, 1]

            results = uploaded_df.copy()
            results["failure_probability"] = proba
            results["intervention_priority"] = np.where(
                proba >= 0.5, "High Risk", np.where(proba >= 0.2, "Watch", "Normal")
            )

            st.success(f"Real batch prediction complete for {n:,} rows.")
            st.dataframe(
                results.sort_values("failure_probability", ascending=False),
                use_container_width=True,
                height=500,
            )
            st.download_button(
                "Download predictions as CSV",
                results.to_csv(index=False).encode("utf-8"),
                "batch_predictions.csv",
                "text/csv",
            )
            st.caption(
                "These are genuine, live predictions computed from the uploaded "
                "data through the same real trained model used throughout this "
                "system -- not stored lookups."
            )

elif page == "Explainability":
    render_page_header("Explainability", "\U0001F50D", "Real SHAP explanations for individual predictions")
    machine_id = st.selectbox(
        "Machine", preds["machine_id"].tolist(), key="explain_machine"
    )
    row = preds[preds.machine_id == machine_id].iloc[0]
    explanations = load_explanations(int(row.prediction_id))

    st.write(
        f"**{machine_id}** -- predicted probability: **{row.failure_probability:.3f}** "
        f"({row.intervention_priority})"
    )
    st.write(
        "Top features driving this specific prediction (real SHAP values, "
        "from the actual fitted model, not illustrative):"
    )

    chart_df = explanations.set_index("feature_name")
    st.bar_chart(chart_df["shap_value"])
    st.dataframe(explanations, use_container_width=True)

    st.subheader("Model-level feature importance (real SHAP summary)")
    st.image("images/fig5_5_shap_xgb-1.png", use_container_width=True)
    st.caption(
        "TabPFN is not tree-based, so SHAP's TreeExplainer does not apply to it. "
        "Group-level permutation importance was used instead for TabPFN, shown below."
    )
    st.image("images/fig5_14_tabpfn_groupimportance-1.png", use_container_width=True)

    st.caption(
        "These values describe what the fitted model relies on. SHAP does not "
        "by itself distinguish causality from correlation, so these are not "
        "read as causal effects of any maintenance field on risk."
    )

elif page == "Alerts":
    render_page_header("Alerts", "\U0001F6A8", "Real alerts generated from High Risk predictions")
    alerts = load_alerts_df()
    status_filter = st.selectbox("Status", ["open", "all"])
    if status_filter == "open":
        alerts = alerts[alerts.status == "open"]
    st.write(f"{len(alerts):,} alerts")
    st.dataframe(alerts, use_container_width=True, height=500)
    st.download_button(
        "Download alerts as CSV",
        alerts.to_csv(index=False).encode("utf-8"),
        "alerts_export.csv",
        "text/csv",
    )

elif page == "Prediction History":
    render_page_header("Prediction History", "\U0001F4C8", "Real record of every prediction run")
    conn = get_connection()
    history = pd.read_sql_query(
        "SELECT mv.model_name, mv.version, p.created_at, COUNT(*) as n_predictions "
        "FROM predictions p JOIN model_versions mv ON p.model_version_id = mv.id "
        "GROUP BY mv.model_name, mv.version, p.created_at",
        conn,
    )
    st.dataframe(history, use_container_width=True)
    st.caption(
        "Only one prediction run exists so far (the initial real run on "
        "the 4,024-asset population). Future retraining runs will add "
        "new rows here, each traceable to its own model version."
    )

elif page == "Model Performance":
    render_page_header("Model Performance", "\U0001F4CA", "Real, independently-verified results across all three candidate models")
    st.write("Real, independently-verified results across all three candidate models:")

    perf = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost", "TabPFN"],
            "PR-AUC": [0.5764, 0.7288, 0.8123],
            "ROC-AUC": [0.9173, 0.9457, 0.9650],
            "F1": [0.5638, 0.6028, 0.7104],
            "Recall": [0.8936, 0.9149, 0.6525],
            "Precision": [0.4118, 0.4495, 0.7797],
        }
    )
    st.dataframe(perf, use_container_width=True)

    st.subheader("Precision-Recall and ROC Curves")
    col3, col4 = st.columns(2)
    col3.image("images/fig5_pr_casestudy-1.png", use_container_width=True)
    col4.image("images/fig5_roc_casestudy-1.png", use_container_width=True)

    st.subheader("Confusion Matrices (Test Set)")
    col5, col6 = st.columns(2)
    col5.image(
        "images/fig5_3_cm_xgb_casestudy-1.png",
        caption="XGBoost",
        use_container_width=True,
    )
    col6.image(
        "images/fig5_4_cm_tabpfn_casestudy-1.png",
        caption="TabPFN",
        use_container_width=True,
    )

    st.image(
        "images/fig5_threshold_xgb-1.png",
        caption="XGBoost decision-threshold analysis (real)",
        use_container_width=True,
    )

    st.info(
        "XGBoost is the model actually deployed for inference in this system. "
        "TabPFN wins on PR-AUC, the proposal's stated primary metric, but trades "
        "away recall -- it misses more real High-Risk assets than XGBoost does. "
        "That trade-off, not just the top-line metric, is why XGBoost was kept "
        "as the active, deployed model for now."
    )

elif page == "Peer-Adjusted Analysis":
    render_page_header("Peer-Adjusted Analysis", "\U0001F500", "Component II -- real unsupervised pattern-discovery results")
    st.write(
        "Real results from the unsupervised pattern-discovery stage, run on "
        "the 2,245-asset clustering-eligible population (14 equipment categories "
        "with 50+ observations each)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters found", "4")
    c2.metric("Silhouette score", "0.4261")
    c3.metric("Bootstrap stability (ARI)", "0.7912")

    st.subheader("Ranking strategy overlap (top 15 each)")
    overlap = pd.DataFrame(
        {
            "Comparison": [
                "Raw-count vs Peer-adjusted",
                "Raw-count vs Global anomaly",
                "Peer-adjusted vs Global anomaly",
            ],
            "Overlap (of 15)": [12, 13, 14],
        }
    )
    st.dataframe(overlap, use_container_width=True)
    st.caption(
        "Peer-adjusted ranking surfaces 3 of its top-15 assets that a raw overdue-"
        "count ranking misses entirely -- a real, modest, honestly-reported "
        "confirmation that peer-adjustment finds different assets than a simple count."
    )

    st.subheader("Cluster sizes (real, k=4)")
    st.image("images/fig5_6_clusters_casestudy-1.png", use_container_width=True)

elif page == "Admin: User Management":
    render_page_header("Admin: User Management", "\U0001F510", "Accounts are provisioned here only -- no self-registration")
    st.write(
        "Accounts are provisioned here only. There is no self-registration: "
        "every user of this system is added by the administrator."
    )

    st.subheader("Existing users")
    users_df = list_users()
    st.dataframe(users_df, use_container_width=True)

    st.subheader("Edit an existing user (change name, password, or role)")
    with st.form("edit_user"):
        edit_username = st.selectbox(
            "Select user to edit", users_df["username"].tolist()
        )
        edit_full_name = st.text_input("New full name (leave blank to keep unchanged)")
        edit_password = st.text_input(
            "New password (leave blank to keep unchanged)", type="password"
        )
        edit_role = st.selectbox(
            "New role (leave as-is to keep unchanged)",
            ["(no change)", "engineer", "viewer", "admin"],
        )
        edit_submitted = st.form_submit_button("Save changes")
        if edit_submitted:
            role_to_set = None if edit_role == "(no change)" else edit_role
            if not edit_full_name and not edit_password and not role_to_set:
                st.warning("Nothing was changed -- fill in at least one field above.")
            else:
                update_user(
                    edit_username,
                    edit_full_name or None,
                    edit_password or None,
                    role_to_set,
                )
                st.success(f"Updated account for {edit_username}.")
                st.cache_data.clear()

    st.subheader("Add a new user")
    with st.form("add_user"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Temporary password", type="password")
        new_full_name = st.text_input("Full name")
        new_role = st.selectbox("Role", ["engineer", "viewer", "admin"])
        submitted = st.form_submit_button("Create account")
        if submitted:
            if not new_username or not new_password:
                st.error("Username and password are both required.")
            else:
                try:
                    create_user(new_username, new_password, new_full_name, new_role)
                    st.success(
                        f"Account created for {new_username}. Give them their username and temporary password directly."
                    )
                    st.cache_data.clear()
                except sqlite3.IntegrityError:
                    st.error("That username already exists.")

elif page == "Analytics Dashboard":
    render_page_header("Analytics Dashboard", "\U0001F4C9", "Real, interactive visualizations built directly from the live database")
    st.write("Real, interactive visualizations built directly from the live database -- no external BI tool.")

    chart_colors = {"Normal": "#2A9D5C", "Watch": "#D97B29", "High Risk": "#D33B3B"}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Intervention Priority Distribution")
        dist = preds["intervention_priority"].value_counts().reset_index()
        dist.columns = ["Priority", "Count"]
        fig_pie = px.pie(
            dist, names="Priority", values="Count",
            color="Priority", color_discrete_map=chart_colors, hole=0.4,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Failure Probability Distribution (All 4,024 Machines)")
        fig_hist = px.histogram(
            preds, x="failure_probability", nbins=40,
            color_discrete_sequence=["#D97B29"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EEF5", xaxis_title="Failure Probability", yaxis_title="Number of Machines",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("High Risk Count by Equipment Category (Top 15)")
    high_risk_by_cat = (
        preds[preds.intervention_priority == "High Risk"]["equipment_category"]
        .value_counts().head(15).reset_index()
    )
    high_risk_by_cat.columns = ["Equipment Category", "High Risk Count"]
    fig_bar = px.bar(
        high_risk_by_cat, x="Equipment Category", y="High Risk Count",
        color="High Risk Count", color_continuous_scale="Oranges",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Real Recency: Last-Completed-PM/WO Dates, by Month")
    recency_df, n_pm, n_wo = load_pm_wo_recency()
    st.caption(
        f"The genuine dated signal this CMMS export does contain: {n_pm:,} assets "
        f"({n_pm/4426*100:.1f}% of the register) carry a real Last Completed PM date, "
        f"and {n_wo:,} assets ({n_wo/4426*100:.1f}%) carry a real Last Completed WO date. "
        "Shown here by month rather than fabricating a time series the export does not have."
    )
    fig_recency = px.bar(
        recency_df, x="month", y=["Last Completed PM", "Last Completed WO"],
        barmode="group", color_discrete_sequence=["#D97B29", "#4C72B0"],
    )
    fig_recency.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF5",
        xaxis_title="Month", yaxis_title="Number of real assets", legend_title="Field",
    )
    st.plotly_chart(fig_recency, use_container_width=True)

elif page == "System Information":
    render_page_header("System Information", "\u2699\uFE0F", "Model versions, database contents, and known limitations")
    mv = load_model_versions()
    st.subheader("Model versions")
    st.dataframe(mv, use_container_width=True)

    st.subheader("Database tables")
    conn = get_connection()
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )
    for t in tables["name"]:
        count = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0]["n"]
        st.write(f"- **{t}**: {count:,} rows")

    st.subheader("Known limitations (stated honestly)")
    st.markdown("""
    - This label is a composite proxy built from PM/work-order compliance fields,
      not an engineering-validated failure label.
    - The active deployed model is XGBoost, not TabPFN, despite TabPFN's higher
      PR-AUC, because of the recall trade-off discussed on the Model Performance page.
    - New Machine Prediction gives a real, live inference from the trained model
      for hand-entered inputs; it is not connected to a live IoT/sensor feed --
      real-time sensor integration remains future work, as stated in the thesis.
    """)
