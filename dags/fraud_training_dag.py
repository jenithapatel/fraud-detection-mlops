"""Airflow DAG: ingest -> preprocess -> train -> register.

Owner: Jenitha. Assumes the repo is mounted at /opt/airflow/project and the
package is installed in the Airflow image (`pip install -e /opt/airflow/project`).
"""

from datetime import datetime

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="fraud_training_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,  # triggered manually or by the drift alert
    catchup=False,
    tags=["fraud", "training"],
) as dag:
    ingest = BashOperator(
        task_id="ingest", bash_command=f"cd {PROJECT_DIR} && python -m fraud.data.ingest"
    )
    preprocess = BashOperator(
        task_id="preprocess", bash_command=f"cd {PROJECT_DIR} && python -m fraud.data.preprocess"
    )
    train = BashOperator(
        task_id="train_and_register",
        bash_command=f"cd {PROJECT_DIR} && python -m fraud.models.train",
    )

    ingest >> preprocess >> train
