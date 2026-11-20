# Team Handoff: Fraud Detection MLOps Project

**For: Claude Code sessions of Jenitha, Akshaj, Sameeksha and Devyani.**
Paste this into your Claude Code session (or point it at `HANDOFF.md` in the repo) together with:

> I'm **<your name>**. Read `HANDOFF.md` and `CLAUDE.md`, then help me with my tasks. Start by running the setup steps.

---

## 1. Project in one paragraph

This is the ADSP 32021 (MLOps and Inference Engineering) group final project. We're building an automated pipeline for credit card fraud detection on the Kaggle ULB dataset: 284,807 transactions, 30 features (`Time`, `V1`–`V28`, `Amount`), binary target `Class`, 0.17% fraud. Grading weighs **engineering rigor, reproducibility and system design** over model accuracy. The primary metric is PR-AUC, and the operational metric is recall at 90% precision. The final presentation is a 10-minute slot in **Week 10**, and **each member must state their own engineering contribution.**

## 2. Rubric (8 parts × 5 points) and who covers each

| Rubric item | Lead |
|---|---|
| a. Problem statement and system architecture | Jenitha (slides: everyone) |
| b. Data ingestion and baselines | Jenitha |
| c. Pipeline automation and experiment tracking | Jenitha (Airflow) + Akshaj (MLflow) |
| d. Containerization and deployment | Sameeksha |
| e. Monitoring and drift engineering | Devyani |
| f. GitHub usage and repository structure | Everyone (PRs, reviews, clean commits) |
| g. Challenges and future work | Everyone writes 2–3 bullets for their stage |
| h. Presentation and team collaboration | Everyone |

Repo deliverables: commented code, **check-in and PR logs**, a README that reproduces the whole pipeline, and a dependency file. Slides deliverables: problem/EDA, metric justification, architecture diagram, MLflow screenshot showing algorithm selection, API + monitoring demo/screenshots, drift evidence, repo link.

## 3. What already exists (verified working end to end)

The scaffold has been run end to end on synthetic data: preprocess → train → register `fraud-detector@champion` → serve via FastAPI → baseline traffic → drift simulation → Evidently report detects drift → Prometheus metrics. Lint and 12 tests pass.

```
src/fraud/config.py              shared contract (paths, FEATURES, MODEL_NAME/ALIAS)
src/fraud/data/                  ingest.py, preprocess.py            [Jenitha]
dags/fraud_training_dag.py       Airflow DAG                         [Jenitha]
src/fraud/models/                train.py, metrics.py                [Akshaj]
src/fraud/serving/               app.py, schemas.py, model_loader.py [Sameeksha]
docker/Dockerfile, docker-compose.yml                                [Sameeksha]
src/fraud/monitoring/            simulate_drift.py, send_to_api.py, drift_report.py [Devyani]
monitoring/                      prometheus.yml, alerts.yml          [Devyani]
tests/                           pytest suite (CI)                   [everyone]
.github/                         CI workflow, PR template
```

## 4. Contracts: don't change without a team PR

1. **Schema:** `config.FEATURES` (30 raw columns, in Kaggle order) + `config.TARGET = "Class"`. Scaling happens **inside the model pipeline**, so the API takes raw features.
2. **Data files:** `data/processed/train.csv`, `data/processed/test.csv` (stratified 80/20, `random_state=42`). `test.csv` is never used for training or tuning; it only goes through the API for production validation.
3. **Model registry:** name `fraud-detector`, alias `champion`. The model is logged with `mlflow.sklearn` (must have `predict_proba`) and the model version has tags `threshold` (decision threshold for 90% precision) and `semver`.
4. **API:** `POST /predict` with `{"transactions": [{<30 features>}, ...]}` returns `{"model_version", "threshold", "predictions": [{"fraud_probability", "is_fraud"}]}`. Also `GET /health`, `POST /reload`, `GET /metrics/`.
5. **Prediction log:** every prediction is appended to `data/predictions/predictions.csv` (features + `fraud_probability`, `is_fraud`, `model_version`, `timestamp`). The drift report reads it.

## 5. Setup (each person, once)

```bash
git clone <repo-url> && cd fraud-detection-mlops
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
git config core.hooksPath .githooks                     # strips AI attribution from commits
git config user.name "<Your Name>"                      # your own identity
git config user.email "<your GitHub email>"
dvc pull                                                # data from the shared Drive folder (Google sign-in on first run)
python -m fraud.data.preprocess
ruff check . && pytest -q
```

Docker Desktop is needed for `docker compose up -d --build` (MLflow on :5000, API on :8000, Prometheus on :9090, Grafana on :3000).

## 6. Git rules (all Claude sessions must follow)

- **No AI attribution anywhere.** No `Co-Authored-By: Claude`, no `Claude-Session`, no "Generated with Claude Code" in commits, PR titles/descriptions, comments or docs. Commits use only the human's git identity. This is enforced three ways: `.claude/settings.json` turns attribution off, `.githooks/commit-msg` strips it, and CI fails any PR containing it. Never use `--no-verify`.
- Never push to `main` or force-push. `main` is protected.
- One branch per task: `data/…`, `model/…`, `serving/…`, `monitoring/…`, `docs/…`.
- `git pull` on main before starting, and rebase on `origin/main` if behind.
- Small commits with clear imperative messages. The commit and PR history is graded.
- Open PRs with `gh pr create` using the template. **A human teammate reviews and merges.** Everyone reviews at least one PR a week.
- Shared files (`config.py`, `requirements*.txt`, CI, `docker-compose.yml`): flag changes in the PR description and tag the team.

## 7. Tasks by person

### Jenitha Patel: data, orchestration, repo/CI owner
- [ ] Create the GitHub repo, push the scaffold, add teammates as collaborators.
- [ ] Protect `main`: require a PR, 1 approval and passing CI checks (`attribution`, `test`, `docker`).
- [ ] Set up DVC: `dvc init`, `dvc add data/raw/creditcard.csv`, commit the `.dvc` file, and configure a remote (Google Drive or a shared folder) so others can `dvc pull`.
- [ ] Strengthen `ingest.py` validation: dtypes, nulls, value ranges, and logging of row counts and fraud rate.
- [ ] Short EDA notebook or script for the slides: class imbalance, Amount and Time distributions.
- [ ] Get Airflow running (official `docker compose` or `airflow standalone`), mount the repo at `/opt/airflow/project`, `pip install -e` it, and show the DAG `fraud_training_pipeline` succeeding (screenshot for the slides).
- [ ] Own `requirements*.txt` and the CI workflow.
- [ ] **Done when:** a fresh clone can `dvc pull` and the DAG runs ingest → preprocess → train → register.

### Akshaj Chandwani: modeling and experiment tracking
- [ ] In `src/fraud/models/train.py`, add **XGBoost** (`scale_pos_weight`) and **FLAML AutoML** (`metric="ap"`, time budget) to `candidate_models()`. For FLAML, log `automl.model.estimator`, not the AutoML object.
- [ ] Optional: a class-imbalance comparison (class weights vs. SMOTE via `imbalanced-learn`; add it to requirements through Jenitha).
- [ ] Log params, PR-AUC, recall@P90, threshold, and a PR-curve plot as an artifact for every run.
- [ ] Implement the semantic versioning rule for the `semver` tag (MAJOR = schema change, MINOR = new algorithm, PATCH = retrain).
- [ ] Keep `test.csv` untouched. Model selection uses the validation split carved from train.
- [ ] Screenshot the MLflow comparison view showing algorithm selection (slides).
- [ ] **Done when:** `python -m fraud.models.train` registers the best model as `fraud-detector@champion` with `threshold` and `semver` tags.

### Sameeksha Kini: containerization and deployment
- [ ] Harden `src/fraud/serving/app.py`: request logging, clear 4xx/5xx errors, and a 422 counter metric for schema-violating requests (a drift signal).
- [ ] Make `docker compose up -d --build` reliable: a healthcheck on the `api` service, a health-checked `depends_on` on MLflow, and retry model loading at startup.
- [ ] Add an API test for `/reload` and for a batch of 1,000 rows. Measure p95 latency and put the number in the README.
- [ ] Test the full stack on a clean machine following only the README, and fix the README wherever it's wrong.
- [ ] Screenshot `/docs` and a live `/predict` call (slides).
- [ ] **Done when:** a teammate can clone, `docker compose up`, train, `POST /reload`, and get predictions with no extra steps.

### Devyani Rastogi: monitoring and drift engineering
- [ ] **Baseline validation:** send the clean `test.csv` through the API and record the baseline PR-AUC, fraud rate and drift share. Tune the thresholds in `monitoring/alerts.yml` from that baseline.
- [ ] **Drift scenarios** (`simulate_drift.py`): amount ×10, out-of-range V-features, swapped V1/V2. Add a schema-change scenario (dropped or renamed column, which should be caught as 422s).
- [ ] Push the Evidently drift share to Prometheus (Pushgateway service in `docker-compose.yml`) and add a `DataDriftDetected` alert.
- [ ] Build a Grafana dashboard (provisioned from JSON under `monitoring/grafana/`): request rate, fraud-rate %, probability histogram, p95 latency, drift share, model loaded.
- [ ] Optional: the drift alert triggers the Airflow DAG to retrain (coordinate with Jenitha).
- [ ] Before/after screenshots: Grafana panels, Prometheus alert firing, Evidently HTML report (slides item f, "Drift Analysis").
- [ ] **Done when:** running a drift scenario visibly fires an alert, and the evidence is saved in `docs/`.

### Everyone
- [ ] Keep your row in `docs/contributions.md` current with links to your PRs.
- [ ] Write 2–3 "challenges and future work" bullets for your stage.
- [ ] Build your slides for your stage. One person submits the single team PPT.

## 8. Timeline

| When | Milestone |
|---|---|
| This week | Repo pushed, `main` protected, everyone set up and running tests locally. Data and DVC done. |
| +1 week | Each stage works on its own against real data (serving can use `FRAUD_MODEL_SOURCE=dummy` until a model is registered). |
| End of Week 9 | **Integration:** full run on one machine, from DAG to registry to API to baseline to drift alert. Screenshots captured. |
| Week 10 | Slides finalized, rehearse 10 minutes, presentation. |

## 9. Useful commands

```bash
make up | make data | make train | make baseline | make drift | make report | make test
FRAUD_MODEL_SOURCE=dummy uvicorn fraud.serving.app:app --reload   # API without MLflow
gh pr create --fill                                                # open a PR (then edit)
```
