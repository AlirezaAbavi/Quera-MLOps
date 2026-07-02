# HW3_C image: base image + my app/ on top. No bundle baked in, the init
# container pulls it from MinIO at startup.
# build: docker build -t 185.50.38.163:35000/qbc12-embedder-alireza-abouei:v1 .
# push:  docker push 185.50.38.163:35000/qbc12-embedder-alireza-abouei:v1
FROM 185.50.38.163:35000/qbc12-hw03-base:v1

# base is missing some app deps (psycopg/qdrant-client/minio), install them
USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY app/ /app/app/
RUN chmod +x /app/app/entrypoint.sh

ENV PYTHONPATH=/app \
    PORT=8000

WORKDIR /app
USER app

EXPOSE 8000
ENTRYPOINT ["/app/app/entrypoint.sh"]
