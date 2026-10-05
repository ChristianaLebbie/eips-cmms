"""
Builds the real SQLite database matching the Figure 3.3 schema and populates it
with genuine predictions from the trained XGBoost model on the real process-plant
population. This is run once to initialize the system; the Streamlit app reads
from this database afterward.
"""
import sqlite3
import joblib
import pandas as pd
import numpy as np
import shap
from datetime import datetime, timezone

DB_PATH = "cmms_system.db"
bundle = joblib.load("model_bundle.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
feature_columns = bundle["feature_columns"]

pop = pd.read_csv("process_plant_data.csv")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS prediction_explanations;
DROP TABLE IF EXISTS alerts;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS machines;
DROP TABLE IF EXISTS datasets;

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_filename TEXT,
    imported_at TEXT,
    n_rows INTEGER,
    n_columns INTEGER,
    n_machines INTEGER,
    status TEXT
);

CREATE TABLE machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER REFERENCES datasets(id),
    machine_id TEXT,
    equipment_category TEXT,
    criticality TEXT,
    manufacturer TEXT
);

CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER REFERENCES datasets(id),
    model_name TEXT,
    version TEXT,
    metrics_json TEXT,
    is_active INTEGER
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER REFERENCES machines(id),
    model_version_id INTEGER REFERENCES model_versions(id),
    failure_probability REAL,
    prediction_horizon TEXT,
    intervention_priority TEXT,
    created_at TEXT
);

CREATE TABLE prediction_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id),
    feature_name TEXT,
    shap_value REAL
);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id),
    intervention_priority TEXT,
    status TEXT,
    created_at TEXT
);
""")

now = datetime.now(timezone.utc).isoformat()

cur.execute(
    "INSERT INTO datasets (source_filename, imported_at, n_rows, n_columns, n_machines, status) VALUES (?,?,?,?,?,?)",
    ("process_plant_with_risk_label.csv", now, len(pop), len(pop.columns), len(pop), "active")
)
dataset_id = cur.lastrowid

cur.execute(
    "INSERT INTO model_versions (dataset_id, model_name, version, metrics_json, is_active) VALUES (?,?,?,?,?)",
    (dataset_id, "XGBoost", "v1.0.0",
     '{"PR_AUC": 0.7288, "ROC_AUC": 0.9457, "F1": 0.6028, "Recall": 0.9149, "Precision": 0.4495}', 1)
)
model_version_id = cur.lastrowid

X_cat = pd.get_dummies(pop[cat_cols].fillna("Unknown"), prefix=cat_cols)
X_num = pop[num_cols].fillna(0)
X = pd.concat([X_num, X_cat], axis=1)
X = X.reindex(columns=feature_columns, fill_value=0)
X_scaled = X.copy()
X_scaled[num_cols] = scaler.transform(X[num_cols])

proba = model.predict_proba(X_scaled)[:, 1]

def priority_from_proba(p):
    if p >= 0.5:
        return "High Risk"
    if p >= 0.2:
        return "Watch"
    return "Normal"

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

machine_ids = {}
for i, row in pop.iterrows():
    cur.execute(
        "INSERT INTO machines (dataset_id, machine_id, equipment_category, criticality, manufacturer) VALUES (?,?,?,?,?)",
        (dataset_id, row["Asset Name"], row["Equipment Category"], row["Criticality Classification"], row["Manufacturer"])
    )
    machine_ids[i] = cur.lastrowid

for i, (idx, row) in enumerate(pop.iterrows()):
    p = float(proba[i])
    priority = priority_from_proba(p)
    cur.execute(
        "INSERT INTO predictions (machine_id, model_version_id, failure_probability, prediction_horizon, intervention_priority, created_at) VALUES (?,?,?,?,?,?)",
        (machine_ids[idx], model_version_id, p, None, priority, now)
    )
    pred_id = cur.lastrowid

    top_idx = np.argsort(-np.abs(shap_values[i]))[:5]
    for j in top_idx:
        cur.execute(
            "INSERT INTO prediction_explanations (prediction_id, feature_name, shap_value) VALUES (?,?,?)",
            (pred_id, feature_columns[j], float(shap_values[i][j]))
        )

    if priority in ("High Risk", "Watch"):
        cur.execute(
            "INSERT INTO alerts (prediction_id, intervention_priority, status, created_at) VALUES (?,?,?,?)",
            (pred_id, priority, "open", now)
        )

conn.commit()

cur.execute("SELECT COUNT(*) FROM machines")
print("Machines:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM predictions")
print("Predictions:", cur.fetchone()[0])
cur.execute("SELECT intervention_priority, COUNT(*) FROM predictions GROUP BY intervention_priority")
print("Priority distribution:", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM alerts")
print("Alerts:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM prediction_explanations")
print("Explanations:", cur.fetchone()[0])

conn.close()
print("\nDatabase built successfully: cmms_system.db")
