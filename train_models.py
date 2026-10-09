from huggingface_hub import login

login(token="hf_PASTE_YOUR_TOKEN_HERE")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("process_plant_data.csv")
print(df.shape)
print(df["target"].value_counts())

num_cols = [
    "Total Completed PMs",
    "Total Completed WOs",
    "days_since_last_completed_pm",
    "days_since_last_completed_wo",
    "Total time spent on PMs in minutes",
    "Total time spent on WOs in minutes",
]
cat_cols = ["Equipment Category", "Manufacturer", "Criticality Classification"]

X = df[num_cols + cat_cols]
y = df["target"]
print(X.shape, y.shape)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
print(len(Xtr), len(Xte))
print(ytr.sum(), yte.sum())

pre_lr = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)
lr = Pipeline(
    [
        ("pre", pre_lr),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)
lr.fit(Xtr, ytr)
print("Logistic Regression trained.")

pre_xgb = ColumnTransformer(
    [
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)
xgb_pipe = Pipeline(
    [
        ("pre", pre_xgb),
        (
            "clf",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    ]
)
xgb_pipe.fit(Xtr, ytr)
print("XGBoost trained.")

raw_feats = {
    "Total Completed PMs": df["Total Completed PMs"],
    "Total Completed WOs": df["Total Completed WOs"],
    "Days since last PM": df["days_since_last_completed_pm"],
    "Days since last WO": df["days_since_last_completed_wo"],
    "Time on PMs (min)": df["Total time spent on PMs in minutes"],
    "Time on WOs (min)": df["Total time spent on WOs in minutes"],
}
crit_weight_map = {
    "C1 - High Criticality Equipment": 3,
    "C2 - Medium Criticality Equipment": 2,
    "C3 - Low Criticality Equipment": 1,
    "Unknown": 1,
}
raw_feats["Criticality Weight"] = df["Criticality Classification"].map(crit_weight_map)

train_idx = Xtr.index
z = {}
weights = {}
for name, s in raw_feats.items():
    mu, sigma = s.loc[train_idx].mean(), s.loc[train_idx].std()
    z[name] = (s - mu) / sigma
    weights[name] = np.corrcoef(s.loc[train_idx], y.loc[train_idx])[0, 1]
wri = sum(weights[name] * z[name] for name in raw_feats)

peer_anomaly = df["Global_Anomaly_Score"]
mipf = 0.5 * wri.rank(pct=True) + 0.5 * peer_anomaly.rank(pct=True)
print("MIPF computed. Sample scores:")
print(mipf.head())


def report(name, scores_test, y_test, threshold):
    pred = (scores_test >= threshold).astype(int)
    print(
        f"{name}: PR-AUC={average_precision_score(y_test, scores_test):.4f}  "
        f"ROC-AUC={roc_auc_score(y_test, scores_test):.4f}  "
        f"Recall={recall_score(y_test, pred):.4f}  Precision={precision_score(y_test, pred):.4f}"
    )


report("Logistic Regression", lr.predict_proba(Xte)[:, 1], yte, threshold=0.5)
report("XGBoost", xgb_pipe.predict_proba(Xte)[:, 1], yte, threshold=0.5)

test_idx = Xte.index
mipf_test = mipf.loc[test_idx]
mipf_thresh = mipf.loc[train_idx].quantile(1 - ytr.mean())
report("MIPF (yours)", mipf_test, yte, threshold=mipf_thresh)

from tabpfn import TabPFNClassifier

pre_tab = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=20), cat_cols),
    ]
)
Xtr_t = pre_tab.fit_transform(Xtr).astype("float32")
Xte_t = pre_tab.transform(Xte).astype("float32")

tabpfn_clf = TabPFNClassifier(device="cpu")
tabpfn_clf.fit(Xtr_t, ytr.values)
proba = tabpfn_clf.predict_proba(Xte_t)[:, 1]
report("TabPFN", proba, yte, threshold=0.5)
