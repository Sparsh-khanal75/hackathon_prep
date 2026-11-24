# Iris ML API — model serving with monitoring

A small MLOps practice project built while preparing for a hackathon. It trains a classifier, serves it through a **FastAPI** REST API with a browser dashboard, exposes **Prometheus** metrics, and packages everything in **Docker** for deployment on Render.

The model is deliberately simple (Iris dataset). The point of the project is the serving and monitoring around it, not the model.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikitlearn) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker) ![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus)

## What it does

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Predicts the species from four measurements; returns the class, the model's probability for it, and the latency |
| `GET /ui` | Web form plus live charts (prediction distribution, response time), built with Chart.js |
| `GET /dashboard` | Compact analytics view that refreshes every 3 seconds |
| `GET /stats` | JSON usage statistics: counts per class, average latency, test accuracy, uptime |
| `GET /metrics` | Prometheus text format: predictions per class, total calls, average latency, test accuracy, uptime |
| `GET /health` | Health check for the hosting platform |
| `GET /docs` | Auto-generated OpenAPI docs |

## Model

`model_train.py` trains a Random Forest (100 trees) on the 150-sample Iris dataset with an 80/20 split and saves `iris_model.pkl` plus `model_info.json`. Test accuracy was **100% on 30 held-out samples**. Iris is an easy, tiny dataset, so this number says little about real-world performance.

```bash
python model_train.py
```

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000          # open http://localhost:8000/ui

# Docker
docker build -t iris-api .
docker run -p 8000:8000 iris-api               # PORT defaults to 8000; Render sets its own

# Prometheus (scrapes the API every 5 s)
docker run -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

`prometheus.yml` targets `172.17.0.1:8000` (the Docker bridge IP on Linux). On macOS or Windows, change it to `host.docker.internal:8000`.

Example request:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.6}'
# {"predicted_flower": "versicolor", "confidence": 0.73, "duration": "0.0050s"}
```

## Project structure

```
model_train.py     trains and saves the model + model_info.json
main.py            FastAPI app: prediction, dashboards, stats, Prometheus metrics, health
iris_model.pkl     trained model (scikit-learn 1.3.2)
Dockerfile         container image for Render / local Docker
prometheus.yml     scrape config for a local Prometheus
```

## What I learned and would change

- **Counters live in memory**, so they reset on restart and aren't shared between workers. A production version would use `prometheus-client` counters/histograms (already in `requirements.txt`) and let Prometheus store history.
- **Average latency is a weak metric.** A histogram with p50/p95 shows tail latency, which is what users feel.
- **The model loads from a pickle**, which can execute code. Only load model files you built yourself.
- **Next steps:** tests and a GitHub Actions pipeline, plus input validation ranges.

These ideas are applied in my later project [urlguard](https://github.com/Sparsh-khanal75/malicious-url-detector), which uses real Prometheus histograms, a CI quality gate and a non-root multi-stage Docker image.
