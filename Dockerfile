FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /opt/cdd

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cdd.txt /opt/cdd/requirements-cdd.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /opt/cdd/requirements-cdd.txt

COPY kafka_concept_drift_consumer.py /opt/cdd/kafka_concept_drift_consumer.py

COPY automl_engine/app /opt/alfie_automl_engine/app

ENV AUTOML_ENGINE_REPO=/opt/alfie_automl_engine

RUN useradd --create-home --uid 10001 cdd \
    && chown -R cdd:cdd \
        /opt/cdd \
        /opt/alfie_automl_engine

USER cdd

CMD ["python", "/opt/cdd/kafka_concept_drift_consumer.py"]
