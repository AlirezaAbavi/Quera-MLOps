FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation -e .

CMD ["airbnb-ops", "run"]