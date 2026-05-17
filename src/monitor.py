import os
import pandas as pd
import psycopg2
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Import baru untuk Evidently 0.7.x
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

# ── Config ────────────────────────────────────────────
REPORT_PATH = "reports"

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.columns = [*data.feature_names, "target"]
    return train_test_split(df, test_size=0.2, random_state=42)

def save_drift_to_postgres(drift_detected: bool, drift_share: float):
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "mlops_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "mlops_pass")
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id SERIAL PRIMARY KEY,
            drift_detected BOOLEAN,
            drift_share FLOAT,
            checked_at TIMESTAMP
        )
    """)

    cur.execute("""
        INSERT INTO drift_metrics (drift_detected, drift_share, checked_at)
        VALUES (%s, %s, %s)
    """, (drift_detected, float(drift_share), datetime.now()))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Drift metrics tersimpan ke PostgreSQL")

def monitor():
    os.makedirs(REPORT_PATH, exist_ok=True)

    train_df, test_df = load_data()

    reference = train_df.drop(columns=["target"])
    current   = test_df.drop(columns=["target"])

    definition  = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    report_file = f"{REPORT_PATH}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    result.save_html(report_file)
    print(f"✅ Report tersimpan di {report_file}")

    result_dict    = result.dump_dict()
    metric_key     = list(result_dict["metric_results"].keys())[0]
    widgets        = result_dict["metric_results"][metric_key]["widget"]
    drift_share    = float(widgets[1]["params"]["counters"][0]["value"])
    drift_detected = drift_share > 0.5

    print(f"{'⚠️  Drift terdeteksi!' if drift_detected else '✅ Tidak ada drift'}")
    print(f"📊 Drift share: {drift_share:.2%}")

    # Hanya simpan ke PostgreSQL kalau bukan di CI
    is_ci = os.getenv("CI", "false").lower() == "true"
    if not is_ci:
        save_drift_to_postgres(drift_detected, drift_share)
        print("✅ Drift metrics tersimpan ke PostgreSQL")
    else:
        print("ℹ️  CI environment — skip save ke PostgreSQL")

if __name__ == "__main__":
    monitor()