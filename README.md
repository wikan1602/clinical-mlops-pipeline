# Multimodal MLOps Pipeline for Clinical Risk Assessment

An enterprise-grade, end-to-end MLOps pipeline designed to predict 30-day post-discharge mortality risk using the MIMIC-IV database. This project implements a **multimodal data fusion approach**, combining structured clinical metrics with unstructured clinical narratives parsed via a local Edge-AI model.

---

## 🏗️ System Architecture & Container Topology

The entire infrastructure runs containerized inside an **Ubuntu/WSL2** microservices architecture managed via Docker Compose.

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Orchestration** | Apache Airflow | Controls DAG dependency graphs and scheduled batch execution loops. |
| **Experiment Tracking** | MLflow | Tracks model metrics, hyperparameter profiles, and registers binaries. |
| **Data Warehouse** | PostgreSQL | Stores clinical features, predictions, and model auditing metadata. |
| **Object Storage** | MinIO | S3-compatible storage used to persist model artifacts and binaries. |
| **Visualization** | Apache Superset | Serves active production risk triage dashboards. |
| **Inference Gateway** | FastAPI / Uvicorn | High-performance endpoint serving real-time risk classification. |

### 🔌 Edge-AI Compute Handshake
To eliminate cloud GPU dependency costs, LLM processing is offloaded to an external **ASUS ROG Ally** acting as a local edge compute node:
* **Model:** `medgemma-4b-it` (quantized for local edge performance).
* **Gateway:** LM-Studio serving an OpenAI-compliant API over LAN.
* **Bridge:** Handled via a direct secure API client request loop from the Airflow scheduler container.

---

## 🔄 Automated Data Pipeline (Airflow DAG)

The workflow executes a sequential directed acyclic graph (DAG) divided into four core operational blocks:

1. **Preprocess Data (`preprocess_data`):** Loads admissions data in memory-efficient 5,000-row chunks to prevent Out-of-Memory (OOM) failures under a 16GB RAM limit. Formulates real Length of Stay (LOS) and isolates true 30-day post-discharge mortality instances.
2. **LLM Feature Extraction (`llm_extraction`):** Loops over unstructured data with a fault-tolerant progress checkpoint architecture. Parses text chunks to extract binary social and clinical markers (`polypharmacy`, `has_social_support`, `needs_rehab_or_nursing`, `followup_scheduled`).
3. **Train XGBoost (`train_xgboost`):** Addresses a severe 3.1% class imbalance using dynamic instance re-weighting (`scale_pos_weight`). Runs 5-Fold Stratified Cross-Validation and optimizes decision thresholds using a Precision-Recall curve.
4. **Export to DB (`export_to_superset`):** Executes high-speed bulk migrations to the PostgreSQL database using optimized `psycopg2` `execute_values` batch streams.

---

## 🕵️‍♂️ Data Hygiene & The Data Leakage Fix

Initial development testing yielded an impossible **0.99 AUC** score. Deep diagnostic audits traced this performance spike to **Target Data Leakage** caused by the text context window:
* **The Problem:** The pipeline was parsing the final 4,000 characters of discharge summaries, which inadvertently contained the doctor's final administrative verdicts (e.g., *"Discharge Disposition: Expired"* or *"Discharged to Hospice Care"*).
* **The Remediation:** Shifted the context window window strictly to the **initial 4,000 characters (`[:4000]`)**, capturing exclusively the *History of Present Illness* and *Chief Complaint*. This eliminated downstream leakage and forced the model to learn early diagnostic risk markers.

---

## 📊 Final Performance Metrics & Multimodal Lift

By comparing a structured-only tabular profile against the integrated multimodal profile, the project validated a massive operational **feature lift**:

* **Tabular Baseline (Structured EHR Only):** Cross-Validation AUC of **~0.70 - 0.75**
* **Multimodal Fusion Model (Structured + LLM Features):** Cross-Validation AUC of **0.9198**

### 🎯 Optimized Model Profile
* **Area Under ROC (AUC):** `0.9198`
* **Recall (Sensitivity):** `0.9091` (Catches 90.9% of true high-risk instances)
* **Precision:** `0.3846` (Highly viable for hospital clinical screening and preventative resource allocation)
* **F1-Score:** `0.5405`

---

## 🚀 How to Run Locally

### Prerequisites
* Docker & Docker Compose
* Ubuntu WSL2 Environment
* LM-Studio hosting an OpenAI-compliant server

### Setup Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/wikan1602/mlflow-stack.git](https://github.com/wikan1602/mlflow-stack.git)
   cd mlflow-stack
   ```
2. Place your raw MIMIC-IV datasets inside the /data directories as detailed in the pipeline layout.
3. Launch the container orchestration network:
    ``` bash
    docker compose up -d --build
    ```
4. Access the platforms:
* **Airflow UI:** localhost:8080
* **MLflow Tracking:** localhost:5000
* **Superset UI:** localhost:8088
