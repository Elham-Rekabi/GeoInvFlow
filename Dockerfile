FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    MPLBACKEND=Agg

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CUDA-enabled PyTorch also supports CPU execution. Use cpu for a smaller image.
ARG TORCH_VARIANT=cu128
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && python -m pip install -r /tmp/requirements-docker.txt \
    && useradd --create-home --uid 1000 researcher \
    && chown researcher:researcher /workspace

COPY --chown=researcher:researcher . /workspace
USER researcher

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace"]
