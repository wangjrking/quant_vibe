FROM python:3.10-slim

ARG PIP_INDEX_URL=
ARG INSTALL_TORCH=false
ARG INSTALL_GM=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QUANT_DATA_DIR=/app/data_file \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        fonts-noto-cjk \
        libgomp1 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --index-url "$PIP_INDEX_URL" -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi \
    && if [ "$INSTALL_TORCH" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && if [ "$INSTALL_GM" = "true" ]; then \
        pip install gm; \
    fi

COPY . .

RUN mkdir -p /app/data_file /app/log /app/logs /app/juejin_strategies

VOLUME ["/app/data_file", "/app/log", "/app/logs", "/app/juejin_strategies"]

CMD ["python", "main.py"]
