FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt && groupadd --system trainer \
    && useradd --system --gid trainer --no-create-home trainer

COPY --chown=trainer:trainer training_mock.py /app/training_mock.py
COPY --chown=trainer:trainer prometheus_metrics.py /app/prometheus_metrics.py

USER trainer

ENTRYPOINT ["python", "/app/training_mock.py"]
