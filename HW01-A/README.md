# HW01-A Dockerized Airbnb Ops Package

Run locally:

```bash
python -m pip install -e .
airbnb-ops run
```

Run with Docker:

```bash
docker compose build
docker compose run --rm airbnb-ops
```

Run with DVC:

```bash
dvc repro
```
