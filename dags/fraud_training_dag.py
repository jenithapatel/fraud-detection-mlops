"""Airflow DAG: ingest -> preprocess -> train -> register.

Owner: Jenitha. Runs in the `airflow` service of docker-compose.yml
(`docker compose --profile airflow up -d --build airflow`), where the repo is
mounted at /opt/airflow/project and the project's dependencies live in a
separate virtualenv whose interpreter is given by FRAUD_PYTHON.
"""

import os
from datetime import datetime

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

PROJECT_DIR = os.getenv("FRAUD_PROJECT_DIR", "/opt/airflow/project")
# Project venv interpreter, kept apart from Airflow's own packages to avoid
# dependency conflicts. Falls back to the current python outside Docker.
PYTHON = os.getenv("FRAUD_PYTHON", "python")


def module_command(module: str) -> str:
    """Bash command that runs `python -m <module>` from the project root."""
    return f"cd {PROJECT_DIR} && {PYTHON} -m {module}"


with DAG(
    dag_id="fraud_training_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,  # triggered manually or by the drift alert
    catchup=False,
    tags=["fraud", "training"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=module_command("fraud.data.ingest"))
    preprocess = BashOperator(
        task_id="preprocess", bash_command=module_command("fraud.data.preprocess")
    )
    train = BashOperator(
        task_id="train_and_register", bash_command=module_command("fraud.models.train")
    )

    ingest >> preprocess >> train
