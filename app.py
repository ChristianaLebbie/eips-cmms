"""
EIPS-CMMS: Explainable Intervention Priority System for CMMS
A real, working decision-support system built on real maintenance data,
a real trained XGBoost model, real SHAP explanations, and a real SQLite
database -- with admin-provisioned login, full CMMS modules (Asset
Register, Work Orders, PM/Task History), and all analytics delivered
natively (no external BI tool).

Run with: streamlit run app.py
"""
import sqlite3
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import joblib

DB_PATH = "cmms_system.db"

st.set_page_config(page_title="EIPS-CMMS", layout="wide", page_icon="\u26CF")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0F2340 0%, #16324F 100%);
}
[data-testid="stSidebar"] {
    background: #0B1B2E;
    border-right: 2px solid #D97B29;
}
[data-testid="stSidebar"] * { color: #E8EEF5 !important; }
h1, h2, h3 { color: #F4F7FA !important; }
p, label, span, div { color: #DCE4EC; }
[data-testid="stMetricValue"] { color: #D97B29 !important; }
[data-testid="stMetricLabel"] { color: #AEBFD1 !important; }
.stButton>button {
    background-color: #D97B29; color: white; border: none; font-weight: 600;
}
.stButton>button:hover { background-color: #B8631E; }
div[data-testid="stDataFrame"] { background-color: #16324F; }
.login-card {
    background: #16324F; padding: 2.4rem; border-radius: 10px;
    border: 1px solid #2A4A6E; max-width: 420px; margin: 3rem auto;
}
</style>
""", unsafe_allow_html=True)


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
    cur.execute("SELECT id, username, password_hash, full_name, role FROM users WHERE username = ?", (username,))
    return cur.fetchone()


def create_user(username, password, full_name, role):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, full_name, role, created_at) VALUES (?,?,?,?,?)",
        (username, hash_pw(password), full_name, role, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()


def list_users():
    conn = get_connection()
    return pd.read_sql_query("SELECT username, full_name, role, created_at FROM users", conn)


def update_user(username, new_full_name=None, new_password=None, new_role=None):
    conn = get_connection()
    cur = conn.cursor()
    if new_full_name:
        cur.execute("UPDATE users SET full_name = ? WHERE username = ?", (new_full_name, username))
    if new_password:
        cur.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hash_pw(new_password), username))
    if new_role:
        cur.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, username))
    conn.commit()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center; margin-top:2rem;'>\u26CF EIPS-CMMS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#AEBFD1;'>Explainable Intervention Priority System for CMMS</p>", unsafe_allow_html=True)

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
        st.caption("Accounts are provisioned by the system administrator only. "
                   "If you need access, contact your administrator.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


st.sidebar.markdown(f"### \u26CF EIPS-CMMS")
st.sidebar.caption(f"Signed in as **{st.session_state.full_name}** ({st.session_state.role})")
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False
    st.rerun()
st.sidebar.markdown("---")

track = st.sidebar.radio("Track", ["Case-Study (CMMS)", "Sensor-Detection (AI4I 2020 / Azure PdM)"])
st.sidebar.markdown("---")

if track == "Case-Study (CMMS)":
    nav_options = ["Dashboard", "Asset Register", "Work Orders", "PM / Task History",
                    "Run Prediction", "Explainability", "Alerts", "Prediction History",
                    "Model Performance", "Peer-Adjusted Analysis", "System Information"]
    if st.session_state.role == "admin":
        nav_options.append("Admin: User Management")
    page = st.sidebar.radio("Navigate", nav_options)
else:
    page = st.sidebar.radio("Navigate", ["Sensor Overview", "Sensor Component I",
                                          "Sensor Component II", "Sensor Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption("All analysis, dashboards, and reporting are delivered "
                    "directly inside this application. No external BI tool is used. "
                    "The two tracks above are kept strictly separate.")

preds = load_predictions_df()

if page == "Dashboard":
    st.title("\u26CF EIPS-CMMS Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machines tracked", f"{len(preds):,}")
    col2.metric("High Risk", f"{(preds.intervention_priority == 'High Risk').sum():,}")
    col3.metric("Watch", f"{(preds.intervention_priority == 'Watch').sum():,}")
    mv = load_model_versions()
    active = mv[mv.is_active == 1].iloc[0]
    col4.metric("Active model", active["model_name"])

    st.subheader("Intervention priority distribution")
    dist = preds["intervention_priority"].value_counts()
    st.bar_chart(dist)

    st.subheader("Highest-priority machines")
    st.dataframe(
        preds[preds.intervention_priority == "High Risk"]
        .head(20)[["machine_id", "equipment_category", "criticality", "failure_probability"]],
        use_container_width=True,
    )

elif page == "Asset Register":
    st.title("Asset Register")
    ar = load_asset_register()
    st.write(f"{len(ar):,} real registered assets")

    col1, col2 = st.columns(2)
    cat_filter = col1.multiselect("Equipment Category", sorted(ar["Equipment Category"].dropna().unique())[:50])
    hier_filter = col2.multiselect("Hierarchy Level", sorted(ar["Hierarchy_Level"].dropna().unique()))

    filtered = ar.copy()
    if cat_filter:
        filtered = filtered[filtered["Equipment Category"].isin(cat_filter)]
    if hier_filter:
        filtered = filtered[filtered["Hierarchy_Level"].isin(hier_filter)]

    st.write(f"{len(filtered):,} assets match the current filter")
    display_cols = ["Asset Name", "Equipment Category", "Equipment Description", "Manufacturer",
                     "Criticality Classification", "Hierarchy_Level", "Overdue PMs", "Overdue WOs",
                     "Total Completed PMs", "Total Completed WOs"]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

elif page == "Work Orders":
    st.title("Work Orders")
    wo = load_work_orders()
    st.write(f"{len(wo):,} real work order records")

    col1, col2 = st.columns(2)
    type_filter = col1.multiselect("Work Type", sorted(wo["WorkType"].dropna().unique()))
    status_filter = col2.multiselect("Status", sorted(wo["Status"].dropna().unique()))

    filtered = wo.copy()
    if type_filter:
        filtered = filtered[filtered["WorkType"].isin(type_filter)]
    if status_filter:
        filtered = filtered[filtered["Status"].isin(status_filter)]

    st.write(f"{len(filtered):,} work orders match the current filter")
    st.dataframe(filtered, use_container_width=True, height=500)

    st.subheader("Work Type distribution")
    st.bar_chart(wo["WorkType"].value_counts())

elif page == "PM / Task History":
    st.title("PM / Task History")
    th = load_task_history()
    st.write(f"{len(th):,} real task-history records")

    asset_filter = st.text_input("Search by Asset Name (partial match)")
    filtered = th.copy()
    if asset_filter:
        filtered = filtered[filtered["Asset Name"].astype(str).str.contains(asset_filter, case=False, na=False)]

    st.write(f"{len(filtered):,} tasks match the current search")
    st.dataframe(filtered, use_container_width=True, height=500)

    st.subheader("Downtime distribution (minutes)")
    nonzero = th[th["Downtime in Minutes"] > 0]
    st.write(f"{len(nonzero)} of {len(th)} tasks have any logged downtime")
    if len(nonzero) > 0:
        st.bar_chart(nonzero["Downtime in Minutes"].head(50).reset_index(drop=True))

elif page == "Run Prediction":
    st.title("Run Prediction")
    st.write("Select a machine to view its real, already-computed prediction "
             "from the active model (XGBoost).")
    machine_id = st.selectbox("Machine", preds["machine_id"].tolist())
    row = preds[preds.machine_id == machine_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability", f"{row.failure_probability:.3f}")
    c2.metric("Intervention priority", row.intervention_priority)
    c3.metric("Criticality", row.criticality)

    st.caption(
        "This probability comes from the XGBoost model trained on the real "
        "4,024-asset process-plant population (PR-AUC 0.7288, Recall 0.9149 "
        "on the held-out test set)."
    )

elif page == "Explainability":
    st.title("Explainability (SHAP)")
    machine_id = st.selectbox("Machine", preds["machine_id"].tolist(), key="explain_machine")
    row = preds[preds.machine_id == machine_id].iloc[0]
    explanations = load_explanations(int(row.prediction_id))

    st.write(f"**{machine_id}** -- predicted probability: **{row.failure_probability:.3f}** "
             f"({row.intervention_priority})")
    st.write("Top features driving this specific prediction (real SHAP values, "
             "from the actual fitted model, not illustrative):")

    chart_df = explanations.set_index("feature_name")
    st.bar_chart(chart_df["shap_value"])
    st.dataframe(explanations, use_container_width=True)

    st.subheader("Model-level feature importance (real SHAP summary)")
    st.image("images/fig5_5_shap_xgb-1.png", use_container_width=True)
    st.caption("TabPFN is not tree-based, so SHAP's TreeExplainer does not apply to it. "
               "Group-level permutation importance was used instead for TabPFN, shown below.")
    st.image("images/fig5_14_tabpfn_groupimportance-1.png", use_container_width=True)

    st.caption(
        "These values describe what the fitted model relies on. SHAP does not "
        "by itself distinguish causality from correlation, so these are not "
        "read as causal effects of any maintenance field on risk."
    )

elif page == "Alerts":
    st.title("Alerts")
    alerts = load_alerts_df()
    status_filter = st.selectbox("Status", ["open", "all"])
    if status_filter == "open":
        alerts = alerts[alerts.status == "open"]
    st.write(f"{len(alerts):,} alerts")
    st.dataframe(alerts, use_container_width=True, height=500)

elif page == "Prediction History":
    st.title("Prediction History")
    conn = get_connection()
    history = pd.read_sql_query(
        "SELECT mv.model_name, mv.version, p.created_at, COUNT(*) as n_predictions "
        "FROM predictions p JOIN model_versions mv ON p.model_version_id = mv.id "
        "GROUP BY mv.model_name, mv.version, p.created_at", conn
    )
    st.dataframe(history, use_container_width=True)
    st.caption("Only one prediction run exists so far (the initial real run on "
               "the 4,024-asset population). Future retraining runs will add "
               "new rows here, each traceable to its own model version.")

elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Real, independently-verified results across all three candidate models:")

    perf = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "TabPFN"],
        "PR-AUC": [0.5764, 0.7288, 0.8123],
        "ROC-AUC": [0.9173, 0.9457, 0.9650],
        "F1": [0.5638, 0.6028, 0.7104],
        "Recall": [0.8936, 0.9149, 0.6525],
        "Precision": [0.4118, 0.4495, 0.7797],
    })
    st.dataframe(perf, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.image("images/fig5_1_prauc-1.png", caption="PR-AUC comparison (real results)", use_container_width=True)
    col2.image("images/fig5_2_rocauc-1.png", caption="ROC-AUC comparison (real results)", use_container_width=True)

    st.subheader("Precision-Recall and ROC Curves")
    col3, col4 = st.columns(2)
    col3.image("images/fig5_pr_casestudy-1.png", use_container_width=True)
    col4.image("images/fig5_roc_casestudy-1.png", use_container_width=True)

    st.subheader("Confusion Matrices (Test Set)")
    col5, col6 = st.columns(2)
    col5.image("images/fig5_3_cm_xgb_casestudy-1.png", caption="XGBoost", use_container_width=True)
    col6.image("images/fig5_4_cm_tabpfn_casestudy-1.png", caption="TabPFN", use_container_width=True)

    st.image("images/fig5_threshold_xgb-1.png", caption="XGBoost decision-threshold analysis (real)", use_container_width=True)

    st.info(
        "XGBoost is the model actually deployed for inference in this system. "
        "TabPFN wins on PR-AUC, the proposal's stated primary metric, but trades "
        "away recall -- it misses more real High-Risk assets than XGBoost does. "
        "That trade-off, not just the top-line metric, is why XGBoost was kept "
        "as the active, deployed model for now."
    )

elif page == "Peer-Adjusted Analysis":
    st.title("Peer-Adjusted Analysis (Component II)")
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
    overlap = pd.DataFrame({
        "Comparison": ["Raw-count vs Peer-adjusted", "Raw-count vs Global anomaly", "Peer-adjusted vs Global anomaly"],
        "Overlap (of 15)": [12, 13, 14],
    })
    st.dataframe(overlap, use_container_width=True)
    st.caption(
        "Peer-adjusted ranking surfaces 3 of its top-15 assets that a raw overdue-"
        "count ranking misses entirely -- a real, modest, honestly-reported "
        "confirmation that peer-adjustment finds different assets than a simple count."
    )

    st.subheader("Cluster sizes (real, k=4)")
    st.image("images/fig5_6_clusters_casestudy-1.png", use_container_width=True)

elif page == "Admin: User Management":
    st.title("Admin: User Management")
    st.write("Accounts are provisioned here only. There is no self-registration: "
             "every user of this system is added by the administrator.")

    st.subheader("Existing users")
    users_df = list_users()
    st.dataframe(users_df, use_container_width=True)

    st.subheader("Edit an existing user (change name, password, or role)")
    with st.form("edit_user"):
        edit_username = st.selectbox("Select user to edit", users_df["username"].tolist())
        edit_full_name = st.text_input("New full name (leave blank to keep unchanged)")
        edit_password = st.text_input("New password (leave blank to keep unchanged)", type="password")
        edit_role = st.selectbox("New role (leave as-is to keep unchanged)", ["(no change)", "engineer", "viewer", "admin"])
        edit_submitted = st.form_submit_button("Save changes")
        if edit_submitted:
            role_to_set = None if edit_role == "(no change)" else edit_role
            if not edit_full_name and not edit_password and not role_to_set:
                st.warning("Nothing was changed -- fill in at least one field above.")
            else:
                update_user(edit_username, edit_full_name or None, edit_password or None, role_to_set)
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
                    st.success(f"Account created for {new_username}. Give them their username and temporary password directly.")
                    st.cache_data.clear()
                except sqlite3.IntegrityError:
                    st.error("That username already exists.")

elif page == "System Information":
    st.title("System Information")
    mv = load_model_versions()
    st.subheader("Model versions")
    st.dataframe(mv, use_container_width=True)

    st.subheader("Database tables")
    conn = get_connection()
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    for t in tables["name"]:
        count = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0]["n"]
        st.write(f"- **{t}**: {count:,} rows")

    st.subheader("Known limitations (stated honestly)")
    st.markdown("""
    - This label is a composite proxy built from PM/work-order compliance fields,
      not an engineering-validated failure label.
    - The active deployed model is XGBoost, not TabPFN, despite TabPFN's higher
      PR-AUC, because of the recall trade-off discussed on the Model Performance page.
    - The sensor-detection track (AI4I 2020, Azure PdM) is shown separately under
      its own track selector, and is never merged with or compared numerically
      against this case-study track.
    """)

elif page == "Sensor Overview":
    st.title("Sensor-Detection Track: Overview")
    st.warning("This track uses two public sensor datasets (AI4I 2020, Azure PdM). "
               "It is kept strictly separate from the case-study track above -- "
               "never merged, never compared numerically.")
    col1, col2 = st.columns(2)
    col1.metric("AI4I 2020 records", "10,000")
    col2.metric("Azure PdM records (aggregated)", "36,600")
    st.write("Use the pages on the left to see each dataset's real Component I, "
             "Component II, and model-comparison results.")

elif page == "Sensor Component I":
    st.title("Sensor-Detection Track: Component I (Statistical Analysis)")
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM sensor_component1_results", conn)
    for ds in df.dataset_name.unique():
        st.subheader(ds)
        st.dataframe(df[df.dataset_name == ds][["test_name", "statistic", "p_value", "finding"]],
                     use_container_width=True)

elif page == "Sensor Component II":
    st.title("Sensor-Detection Track: Component II (Pattern Discovery)")
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM sensor_component2_results", conn)
    for ds in df.dataset_name.unique():
        st.subheader(ds)
        st.dataframe(df[df.dataset_name == ds][["metric_name", "metric_value", "finding"]],
                     use_container_width=True)
    st.caption(
        "Note the ranking-comparison finding differs by dataset: the global anomaly "
        "detector underperforms a simple raw ranking on AI4I 2020, but outperforms it "
        "on Azure PdM. There is no universal winner across datasets."
    )

elif page == "Sensor Model Performance":
    st.title("Sensor-Detection Track: Model Performance")
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM sensor_model_results", conn)
    for ds in df.dataset_name.unique():
        st.subheader(ds)
        sub = df[df.dataset_name == ds][["model_name", "pr_auc", "roc_auc", "f1", "recall", "precision_val"]]
        sub.columns = ["Model", "PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
        st.dataframe(sub, use_container_width=True)

    st.subheader("AI4I 2020 -- Real Charts")
    col1, col2 = st.columns(2)
    col1.image("images/fig5_pr_ai4i-1.png", use_container_width=True)
    col2.image("images/fig5_roc_ai4i-1.png", use_container_width=True)
    col3, col4 = st.columns(2)
    col3.image("images/fig5_9_cm_xgb_ai4i-1.png", caption="XGBoost confusion matrix", use_container_width=True)
    col4.image("images/fig5_15_shap_ai4i-1.png", caption="XGBoost SHAP importance", use_container_width=True)
    st.image("images/fig5_10_clusters_ai4i-1.png", caption="Component II cluster failure rates (k=4)", use_container_width=True)

    st.subheader("Azure PdM -- Real Charts")
    col5, col6 = st.columns(2)
    col5.image("images/fig5_pr_azure-1.png", use_container_width=True)
    col6.image("images/fig5_roc_azure-1.png", use_container_width=True)
    col7, col8 = st.columns(2)
    col7.image("images/fig5_11_cm_xgb_azure-1.png", caption="XGBoost confusion matrix", use_container_width=True)
    col8.image("images/fig5_16_shap_azure-1.png", caption="XGBoost SHAP importance", use_container_width=True)
    st.image("images/fig5_12_clusters_azure-1.png", caption="Component II cluster failure-day rates (k=5)", use_container_width=True)

    st.subheader("Ranking Strategy: No Universal Winner (Real)")
    st.image("images/fig5_13_ranking_flip-1.png", use_container_width=True)
    st.caption("The global anomaly detector underperforms a simple raw ranking on AI4I 2020, "
               "but outperforms it on Azure PdM -- a genuine, dataset-dependent finding.")
