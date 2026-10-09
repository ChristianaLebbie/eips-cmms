# EIPS-CMMS (Explainable Intervention Priority System for CMMS)

A real, working CMMS-style decision-support system: real Asset Register,
Work Orders, and PM/Task History modules, a real trained XGBoost model,
real SHAP explanations, a real SQLite database, and admin-provisioned
login -- no self-registration.

## How to run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py
   (or: python -m streamlit run app.py, if "streamlit" isn't on PATH)

3. Open the URL it prints (usually http://localhost:8501)

## Logging in

Two seed accounts exist to start:
- Username: admin      Password: admin123     (role: admin)
- Username: engineer   Password: engineer123  (role: engineer)

**Change these before sharing the system with anyone.** As admin, go to
"Admin: User Management" in the sidebar to add every other real user
yourself -- there is no public sign-up.

## Modules

- Dashboard -- KPIs and priority distribution
- Asset Register -- the real 4,426-asset register, searchable/filterable
- Work Orders -- the real 2,392 work order records
- PM / Task History -- the real 7,796 task-history records
- Run Prediction / Explainability / Alerts / Prediction History -- the
  ML decision-support layer (4,024-asset modelling population)
- Model Performance / Peer-Adjusted Analysis -- real Component II/III results
- Admin: User Management -- add/view users (admin role only)
- System Information -- database contents and known limitations
- Sensor-Detection track -- AI4I 2020 and Azure PdM, kept strictly separate
