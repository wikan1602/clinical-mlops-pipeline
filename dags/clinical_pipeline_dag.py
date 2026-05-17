from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from openai import OpenAI
import sys
sys.path.append('/opt/airflow/src')

# ==========================================================
# CONFIGURATION & PATHS (Absolute Container Paths)
# ==========================================================
MIMIC_BASE = '/opt/airflow/data/mimic-iv/'
HOSP_PATH = f'{MIMIC_BASE}hosp/'
NOTE_PATH = f'{MIMIC_BASE}note/'
PROCESSED_PATH = '/opt/airflow/data/processed/'

CHECKPOINT_PATH = f'{PROCESSED_PATH}llm_checkpoints.csv'
FINAL_TRAIN_SET = f'{PROCESSED_PATH}pilot_final_training_set.parquet'

# PILOT_SIZE: Set to a small number (e.g., 10000) for testing 
# to ensure your 16GB RAM stays stable.
PILOT_SIZE = 500

# ==========================================================
# 1. PREPROCESS TASK (Calculates Real LOS & Post-Discharge Mortality)
# ==========================================================
def preprocess_mimic_data():
    print("🚀 Starting Leak-Free Preprocessing...")
    
    adm_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime']
    pat_cols = ['subject_id', 'dod', 'gender', 'anchor_age']
    
    admissions = pd.read_csv(f'{HOSP_PATH}admissions.csv.gz', usecols=adm_cols, nrows=PILOT_SIZE)
    patients = pd.read_csv(f'{HOSP_PATH}patients.csv.gz', usecols=pat_cols)
    
    df = admissions.merge(patients, on='subject_id', how='inner')
    
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['dod'] = pd.to_datetime(df['dod'])
    
    # ✅ FIX: True post-discharge mortality (died after leaving, up to 30 days)
    df['days_to_death'] = (df['dod'] - df['dischtime']).dt.days
    df['mortality_30d'] = ((df['days_to_death'] > 0) & (df['days_to_death'] <= 30)).astype(int)
    
    df['gender_binary'] = df['gender'].map({'M': 1, 'F': 0})
    df.rename(columns={'anchor_age': 'age'}, inplace=True)

    df = df.sort_values(['subject_id', 'admittime'])
    df['prior_admissions'] = df.groupby('subject_id').cumcount()
    
    print("📖 Reading discharge notes in chunks...")
    relevant_ids = set(df['hadm_id'].unique())
    note_cols = ['hadm_id', 'text']
    processed_notes = []

    for chunk in pd.read_csv(f'{NOTE_PATH}discharge.csv.gz', usecols=note_cols, chunksize=5000):
        filtered_chunk = chunk[chunk['hadm_id'].isin(relevant_ids)]
        processed_notes.append(filtered_chunk)
    
    notes = pd.concat(processed_notes)
    df_final = df.merge(notes, on='hadm_id', how='inner')
    
    # ✅ FIX: Real Length of Stay calculation in days
    df_final['los'] = (df_final['dischtime'] - df_final['admittime']).dt.total_seconds() / 86400.0

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df_final.to_parquet(f'{PROCESSED_PATH}structured_features.parquet', index=False)
    print(f"✅ Preprocessing complete. Saved {len(df_final)} records.")

# ==========================================================
# 2. LLM EXTRACTION TASK (Focuses on Early Admission Text)
# ==========================================================
def extract_llm_features():
    print("🧠 Starting Safe LLM Extraction...")
    df_pilot = pd.read_parquet(f'{PROCESSED_PATH}structured_features.parquet')
    client = OpenAI(base_url="http://192.168.0.109:1234/v1", api_key="lm-studio")

    pilot_features = []
    if os.path.exists(CHECKPOINT_PATH):
        pilot_features = pd.read_csv(CHECKPOINT_PATH).to_dict('records')
        processed_ids = [f['hadm_id'] for f in pilot_features]
        df_pilot = df_pilot[~df_pilot['hadm_id'].isin(processed_ids)]

    for idx, record in df_pilot.iterrows():
        try:
            # ✅ FIX: Read early text [:4000] (History/Presentation) to prevent discharge leakage
            safe_text = record['text'][:4000]
            
            response = client.chat.completions.create(
                model="medgemma-4b-it",
                messages=[
                    {"role": "system", "content": "You are a clinical parser. Output valid JSON only."},
                    {"role": "user", "content": f"Based on the initial presentation context, extract as JSON (true/false): has_social_support, needs_rehab_or_nursing, followup_scheduled, polypharmacy. Text: {safe_text}"}
                ],
                temperature=0,
            )
            raw_output = response.choices[0].message.content
            
            start, end = raw_output.find('{'), raw_output.rfind('}')
            if start != -1 and end != -1:
                extracted = json.loads(raw_output[start:end+1])
                for key in ['has_social_support', 'needs_rehab_or_nursing', 'followup_scheduled', 'polypharmacy']:
                    val = extracted.get(key, False)
                    extracted[key] = 1 if val is True or str(val).lower() == 'true' else 0
                
                extracted['hadm_id'] = record['hadm_id']
                pilot_features.append(extracted)
            
            if len(pilot_features) % 10 == 0:
                pd.DataFrame(pilot_features).to_csv(CHECKPOINT_PATH, index=False)

        except Exception as e:
            print(f"Error on admission {record['hadm_id']}: {e}")
            continue

    df_extracted = pd.DataFrame(pilot_features)
    df_final = pd.read_parquet(f'{PROCESSED_PATH}structured_features.parquet')
    df_final = df_final.merge(df_extracted, on='hadm_id', how='inner')
    
    df_final.to_parquet(FINAL_TRAIN_SET, index=False)
    print("✅ LLM feature extraction complete.")

# ==========================================================
# 3. TRAINING TASK (Trains Robust 8-Feature Balanced Model)
# ==========================================================
def train_model():
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve

    print("🚀 Starting Final Multimodal Training...")
    df = pd.read_parquet(FINAL_TRAIN_SET)
    
    # ✅ SIGNATURE SYNC: Restored to all 8 features
    features = ['age', 'los', 'prior_admissions', 'gender_binary', 
                'has_social_support', 'needs_rehab_or_nursing', 
                'followup_scheduled', 'polypharmacy']
    
    X = df[features].fillna(0)
    y = df['mortality_30d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(
        xgb.XGBClassifier(scale_pos_weight=spw, n_estimators=100, max_depth=3, learning_rate=0.1),
        X, y, cv=cv, scoring='roc_auc'
    )
    print(f"📊 Safe CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=spw)
    model.fit(X_train, y_train)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Readmission_Prediction_Pilot")

    try:
        with mlflow.start_run(run_name="Final_Multimodal_Run") as run:
            preds_proba = model.predict_proba(X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, preds_proba)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_threshold = float(thresholds[f1_scores[:-1].argmax()])

            preds_label = (preds_proba >= best_threshold).astype(int)

            mlflow.log_params({"pilot_size": PILOT_SIZE, "scale_pos_weight": round(spw, 4), "best_threshold": round(best_threshold, 4)})
            mlflow.log_metric("auc_roc", roc_auc_score(y_test, preds_proba))
            mlflow.log_metric("precision", precision_score(y_test, preds_label, zero_division=0))
            mlflow.log_metric("recall", recall_score(y_test, preds_label, zero_division=0))
            mlflow.log_metric("f1_score", f1_score(y_test, preds_label, zero_division=0))
            mlflow.log_metric("cv_auc_mean", round(cv_aucs.mean(), 4))
            mlflow.log_metric("cv_auc_std", round(cv_aucs.std(), 4))

            mlflow.sklearn.log_model(model, "model")
            
            with open(f'{PROCESSED_PATH}latest_run_id.txt', 'w') as f:
                f.write(f"{run.info.run_id}\n{best_threshold}")
            print(f"💾 Production artifacts successfully registered in MLflow.")
    except Exception as e:
        print(f"❌ MLFLOW ERROR: {e}"); raise

# ==========================================================
# 4. EXPORT TASK (Exports Full 8-Feature Dataset to Postgres)
# ==========================================================
def push_results_to_postgres():
    import psycopg2
    from psycopg2.extras import execute_values
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

    print("📤 Syncing final results to Postgres warehouse...")
    df = pd.read_parquet(FINAL_TRAIN_SET)

    # ✅ SIGNATURE SYNC: Matches the training signature perfectly
    features = ['age', 'los', 'prior_admissions', 'gender_binary',
                'has_social_support', 'needs_rehab_or_nursing',
                'followup_scheduled', 'polypharmacy']

    mlflow.set_tracking_uri("http://mlflow:5000")
    with open(f'{PROCESSED_PATH}latest_run_id.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        run_id, threshold = lines[0], float(lines[1])

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    X = df[features].fillna(0)
    
    df['predicted_proba'] = model.predict_proba(X)[:, 1]
    df['predicted_label'] = (df['predicted_proba'] >= threshold).astype(int)
    df['risk_tier'] = pd.cut(df['predicted_proba'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High']).astype(str)

    def safe_convert(x):
        if pd.isnull(x): return None
        if isinstance(x, pd.Timestamp): return str(x)
        if isinstance(x, np.generic): return x.item()
        return x

    conn = psycopg2.connect("postgresql://admin:mlops_pass@postgres:5432/mlops_db")
    try:
        cur = conn.cursor()
        pred_cols = ['subject_id', 'hadm_id', 'age', 'gender', 'los', 'prior_admissions', 
                     'has_social_support', 'needs_rehab_or_nursing', 'followup_scheduled', 
                     'polypharmacy', 'mortality_30d', 'predicted_proba', 'predicted_label', 'risk_tier']
        df_pred = df[pred_cols].copy()

        cur.execute("DROP TABLE IF EXISTS clinical_predictions")
        col_defs = ", ".join(f'"{c}" TEXT' for c in df_pred.columns)
        cur.execute(f"CREATE TABLE clinical_predictions ({col_defs})")

        records_pred = [tuple(safe_convert(x) for x in row) for row in df_pred.itertuples(index=False)]
        cols_pred = ", ".join(f'"{c}"' for c in df_pred.columns)
        execute_values(cur, f"INSERT INTO clinical_predictions ({cols_pred}) VALUES %s", records_pred, page_size=500)

        # ✅ FIX: Corrected date string syntax from %Human to %H
        metrics = {
            'run_id': run_id,
            'run_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'pilot_size': str(len(df)),
            'auc_roc': str(round(roc_auc_score(df['mortality_30d'], df['predicted_proba']), 4)),
            'accuracy': str(round(accuracy_score(df['mortality_30d'], df['predicted_label']), 4)),
            'precision': str(round(precision_score(df['mortality_30d'], df['predicted_label'], zero_division=0), 4)),
            'recall': str(round(recall_score(df['mortality_30d'], df['predicted_label'], zero_division=0), 4)),
            'f1_score': str(round(f1_score(df['mortality_30d'], df['predicted_label'], zero_division=0), 4)),
        }

        cur.execute("DROP TABLE IF EXISTS model_metrics")
        cur.execute("CREATE TABLE model_metrics (run_id TEXT, run_time TEXT, pilot_size TEXT, auc_roc TEXT, accuracy TEXT, precision TEXT, recall TEXT, f1_score TEXT)")
        execute_values(cur, "INSERT INTO model_metrics VALUES %s", [tuple(metrics.values())])

        conn.commit()
        print("✅ Pipeline execution completed successfully. Database up to date.")
    except Exception as e:
        conn.rollback(); print(f"❌ Migration failed: {e}"); raise
    finally:
        cur.close(); conn.close()

# ==========================================================
# DAG DEFINITION
# ==========================================================
with DAG(
    'clinical_readmission_pipeline', 
    start_date=datetime(2026, 5, 1), 
    schedule_interval=None,
    catchup=False
) as dag:
    
    t1 = PythonOperator(task_id='preprocess_data', python_callable=preprocess_mimic_data)
    t2 = PythonOperator(task_id='llm_extraction', python_callable=extract_llm_features)
    t3 = PythonOperator(task_id='train_xgboost', python_callable=train_model)
    t4 = PythonOperator(
        task_id='export_to_superset',
        python_callable=push_results_to_postgres
    )

    # 3. Update the Flow
    t1 >> t2 >> t3 >> t4