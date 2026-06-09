# HW02 - FastAPI Model Serving

This project serves the selected machine learning model from Homework 2-B as a FastAPI application.

The goal of this homework is to move the trained model out of the notebook environment and expose it through a reliable API with:

* FastAPI endpoints
* Swagger/OpenAPI documentation
* Pydantic input/output validation
* safe error handling
* leakage-field protection
* clear prediction responses
* Docker support

The model predicts whether a listing is likely to have high demand based on the cleaned feature dataset created in Homework 1 and the selected model from Homework 2.

---

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── schemas.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

---

## Main Components

### `app/main.py`

Defines the FastAPI application and exposes the main API endpoints:

* `GET /`
* `GET /health`
* `POST /predict`

It also controls model loading during startup.

---

### `app/config.py`

Contains the main configuration values, including:

* MLflow tracking URI
* MLflow experiment name
* selected run ID
* expected feature columns
* forbidden leakage fields
* prediction threshold
* model loading settings

Most values can be overridden with environment variables.

---

### `app/model_loader.py`

Responsible for loading the selected model.

The service loads the selected model from MLflow using:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_EXPERIMENT_NAME
MLFLOW_RUN_ID
```

The selected model comes from Homework 2.

---

### `app/schemas.py`

Defines the request and response schemas using Pydantic.

The input schema validates feature types and rejects unexpected fields. This is important because leakage fields, target fields, audit fields, and future-looking fields must not be accepted by the API.

---

### `app/predictor.py`

Converts validated API input into the exact feature format expected by the trained model.

It also:

* maps API-friendly fields to model fields
* derives internal model features
* checks missing features
* blocks forbidden fields
* runs prediction
* returns a clear prediction response

---

## Selected Model

The selected model is the final clean model from Homework 2.

```text
Model: v5_random_forest
Dataset version: v1_student
MLflow experiment: qbc12_hw02_student_alireza_abouei
Selected run ID: 2f4885cc6b2b42baa62c279937ea4401
```

This model was selected because it had the best clean performance among the finished non-leaky runs.

The intentionally leaky model from Homework 2 was not selected, even though it had very high metrics, because it used future information and is not valid for real serving.

---

## Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
MLFLOW_TRACKING_URI=http://185.50.38.163:33014
MLFLOW_TRACKING_USERNAME=your_mlflow_username
MLFLOW_TRACKING_PASSWORD=your_mlflow_password

STUDENT_USERNAME=student_alireza_abouei
MLFLOW_EXPERIMENT_NAME=qbc12_hw02_student_alireza_abouei
MLFLOW_RUN_ID=2f4885cc6b2b42baa62c279937ea4401

PREDICTION_THRESHOLD=0.5
LOCAL_MODEL_PATH=
LOAD_MODEL_ON_STARTUP=true
PORT=8000
```

Do not commit the real `.env` file to Git.

---

## Install Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open Swagger:

```text
http://127.0.0.1:8001/docs
```

Open the health endpoint:

```text
http://127.0.0.1:8001/health
```

---

## Run with Docker

Build and run the service:

```bash
docker compose up --build
```

The container runs FastAPI on port `8000`, and Docker maps it to local port `8001`.

Open Swagger:

```text
http://127.0.0.1:8001/docs
```

Open health check:

```text
http://127.0.0.1:8001/health
```

Stop the service:

```bash
docker compose down
```

---

## Model Download Time

When the API starts, it may need to download the selected model artifact from the MLflow server.

This can take some time depending on:

* MLflow server speed
* network connection
* artifact size
* first-time Docker/container startup

During this time, the application may be running but the model may not be ready for prediction yet.

Check the health endpoint while waiting:

```text
http://127.0.0.1:8001/health
```

If the model is still loading, the response may look like:

```json
{
  "status": "loading",
  "model_loaded": false,
  "error": "Model loading in progress."
}
```

Wait and refresh the health endpoint again.

When the model has loaded successfully, the response should look like:

```json
{
  "status": "ok",
  "model_loaded": true,
  "error": null
}
```

Only use the `/predict` endpoint after `model_loaded` becomes `true`.

If the health endpoint returns an authentication error such as `401 Authorization Required`, check the MLflow username and password in the `.env` file and restart the service.

If the download stays stuck for a long time, check the Docker logs:

```bash
docker compose logs -f
```

---

## API Endpoints

### `GET /`

Basic API information.

Example response:

```json
{
  "message": "HW03 FastAPI model serving is running.",
  "docs_url": "/docs",
  "health_url": "/health"
}
```

---

### `GET /health`

Checks whether the API is running and whether the model is loaded.

Example successful response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "error": null
}
```

If the model is still loading or failed to load, the response will show that state clearly.

---

### `POST /predict`

Runs prediction for one listing.

The endpoint accepts only safe model input fields. It does not accept target fields, leakage fields, future fields, IDs, or audit fields.

---

## Example Prediction Request

```json
{
  "accommodates": 2,
  "available_days_last_30d": 11,
  "available_days_last_90d": 16,
  "available_rate_last_30d": 0.3548,
  "available_rate_last_90d": 0.1758,
  "avg_comment_len_before_cutoff": 302.1672,
  "avg_maximum_nights_calendar_last_30d": 30,
  "avg_maximum_nights_calendar_last_90d": 30,
  "avg_minimum_nights_calendar_last_30d": 3,
  "avg_minimum_nights_calendar_last_90d": 3,
  "bathrooms": 1.5,
  "bedrooms": 1,
  "beds": 1,
  "days_since_last_review": 94,
  "host_is_superhost": true,
  "host_listing_count": 1,
  "instant_bookable": false,
  "listing_price": 132,
  "max_comment_len_before_cutoff": 1917,
  "maximum_nights": 356,
  "minimum_nights": 3,
  "neighbourhood_name": "Centrum-West",
  "property_type": "Private room in houseboat",
  "room_type": "Private room",
  "total_reviews_before_cutoff": 311,
  "unique_reviewers_before_cutoff": 311
}
```

---

## Example Prediction Response

```json
{
  "prediction": 0,
  "prediction_label": "not_high_demand_proxy",
  "probability": 0.250343,
  "threshold": 0.5
}
```

Response fields:

| Field              | Meaning                                      |
| ------------------ | -------------------------------------------- |
| `prediction`       | Binary prediction, `0` or `1`                |
| `prediction_label` | Human-readable label                         |
| `probability`      | Estimated probability for the positive class |
| `threshold`        | Decision threshold used by the API           |

---

## Input Validation

The API validates input with Pydantic.

Invalid input should return a clear error instead of silently producing a prediction.

Examples of invalid input:

* missing required fields
* wrong data types
* negative values for fields that must be non-negative
* invalid categorical values
* extra fields
* target/leakage fields

---

## Leakage Protection

The API must not accept fields that would not be available at prediction time.

Examples of blocked fields:

```text
high_demand_proxy
future_available_days_30d
future_available_rate_30d
target
label
listing_id
host_id
reviewer_id
review_id
license
```

If one of these fields is sent to `/predict`, the API should reject the request.

This prevents accidentally serving a model with future information or target leakage.

---

## Swagger Testing Checklist

Open:

```text
http://127.0.0.1:8001/docs
```

Then test:

### 1. Health endpoint

Run:

```text
GET /health
```

Expected result:

```text
status = ok
model_loaded = true
```

---

### 2. Valid prediction request

Run:

```text
POST /predict
```

with a valid listing payload.

Expected result:

```text
HTTP 200
prediction is 0 or 1
probability is between 0 and 1
threshold is shown
```

---

### 3. Missing field validation

Remove a required field, for example:

```text
listing_price
```

Expected result:

```text
HTTP 422
validation error
```

---

### 4. Wrong type validation

Send a string instead of a number, for example:

```json
{
  "listing_price": "not-a-number"
}
```

Expected result:

```text
HTTP 422
validation error
```

---

### 5. Leakage field rejection

Add a forbidden field, for example:

```json
{
  "future_available_rate_30d": 0.9
}
```

Expected result:

```text
HTTP 422 or 400
request rejected
```

---

## Troubleshooting

### Swagger does not open

Check whether the server is actually running.

For local run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

For Docker:

```bash
docker compose ps
docker compose logs -f
```

---

### `/health` returns model loading error

Check MLflow credentials in `.env`:

```env
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

Restart the app after changing `.env`.

---

### MLflow returns 401

This means the MLflow server rejected the username/password.

Check:

```bash
python - <<'PY'
import os
from dotenv import load_dotenv

load_dotenv()
print("username:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("password_set:", bool(os.getenv("MLFLOW_TRACKING_PASSWORD")))
print("experiment:", os.getenv("MLFLOW_EXPERIMENT_NAME"))
print("run_id:", os.getenv("MLFLOW_RUN_ID"))
PY
```

Do not print the password itself.

---

### Docker cannot read `.env`

Make sure `.env` exists in the same directory as `docker-compose.yml`.

```bash
ls -la .env docker-compose.yml
```

---

### Prediction returns missing columns

This usually means the API input schema and trained model feature list do not match.

The API should internally map:

```text
host_is_superhost -> is_superhost
```

and derive:

```text
has_reviews_before_cutoff
```

from:

```text
total_reviews_before_cutoff
```

---

## Notes

This API is designed for homework and local testing. It demonstrates the required serving workflow:

* loading a selected MLflow model
* validating input with Pydantic
* exposing Swagger documentation
* preventing leakage fields
* returning understandable predictions
* supporting Docker-based execution

The API is not intended as a production deployment without additional security, monitoring, authentication, and deployment hardening.
