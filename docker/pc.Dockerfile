FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates openssh-client zstd rsync \
 && rm -rf /var/lib/apt/lists/*

# Install uv (fast pip)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --yes
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# Preinstall project deps if desired
COPY requirements.txt ./
RUN if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi

# Install as editable to expose console scripts (e.g., meta, export-p8)
COPY pyproject.toml ./
RUN uv pip install -e . || true

ENTRYPOINT ["bash"]

