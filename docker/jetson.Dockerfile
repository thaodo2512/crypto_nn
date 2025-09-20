FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates zstd jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# Install runtime deps: onnxruntime-gpu for Jetson, FastAPI/uvicorn
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      fastapi uvicorn pydantic numpy \
      onnx onnxruntime-gpu==1.16.0 \
      onnxruntime-extensions==0.10.0

# Copy service (expects service_p9.py in repo)
COPY service_p9.py /opt/app/service_p9.py

# Create non-root user
RUN useradd -ms /bin/bash app && chown -R app:app /opt/app
USER app

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -sf http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["python", "/opt/app/service_p9.py"]

