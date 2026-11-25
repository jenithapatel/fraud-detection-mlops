# Airflow image for the training DAG. Build from repo root (docker compose does this).
# The project's dependencies go in their own virtualenv (/home/airflow/fraud-venv)
# so they can't conflict with Airflow's pinned packages. The DAG runs every task
# with that venv's Python (FRAUD_PYTHON), and the repo itself is mounted at
# runtime, not copied, so code changes need no rebuild.
FROM apache/airflow:3.1.3-python3.11

COPY requirements.txt requirements-serving.txt /tmp/requirements/
RUN python -m venv /home/airflow/fraud-venv \
    && /home/airflow/fraud-venv/bin/pip install --no-cache-dir -r /tmp/requirements/requirements.txt
