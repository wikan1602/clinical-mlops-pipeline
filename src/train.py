import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_california_housing  # pengganti Boston (deprecated)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psycopg2
from datetime import datetime


# ── Config ──────────────────────────────────────────
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME     = "boston-housing-xgboost"

# ── Parameter XGBoost ────────────────────────────────
PARAMS = {
    "n_estimators":     100,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
}


def save_metrics_to_postgres(run_id, params, metrics):
    # Skip kalau jalan di CI
    is_ci = os.getenv("CI", "false").lower() == "true"
    if is_ci:
        print("ℹ️  CI environment — skip save ke PostgreSQL")
        return

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "mlops_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "mlops_pass")
    )
    cur = conn.cursor()

    # Buat tabel kalau belum ada
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50),
            experiment_name VARCHAR(100),
            rmse FLOAT,
            mae FLOAT,
            r2 FLOAT,
            n_estimators INT,
            max_depth INT,
            learning_rate FLOAT,
            trained_at TIMESTAMP
        )
    """)

    cur.execute("""
        INSERT INTO model_metrics 
        (run_id, experiment_name, rmse, mae, r2, n_estimators, max_depth, learning_rate, trained_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id,
        EXPERIMENT_NAME,
        float(metrics["rmse"]),      # tambah float()
        float(metrics["mae"]),       # tambah float()
        float(metrics["r2"]),        # tambah float()
        int(params["n_estimators"]), # tambah int()
        int(params["max_depth"]),    # tambah int()
        float(params["learning_rate"]), # tambah float()
        datetime.now()
    ))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Metrics tersimpan ke PostgreSQL")

def load_data():
    data = fetch_california_housing()  # scikit-learn >= 1.0
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(PARAMS)

        # Train
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)

        # Tambah setelah mlflow.log_metrics(metrics)
        save_metrics_to_postgres(
            run_id=mlflow.active_run().info.run_id,
            params=PARAMS,
            metrics=metrics
        )

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"✅ RMSE : {metrics['rmse']:.4f}")
        print(f"✅ MAE  : {metrics['mae']:.4f}")
        print(f"✅ R²   : {metrics['r2']:.4f}")
        print(f"🔗 Run selesai — cek MLflow di {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    train()