# syntax=docker/dockerfile:1.7

ARG RUNTIME_BASE=dependencies
ARG GPU_BASE=gpu

# gpu-image-inputs-begin
ARG PYTHON_IMAGE=python:3.14-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS gpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADLAB_PROJECT_ROOT=/root/gradlab
ENV UV_PROJECT_ENVIRONMENT=/root/gradlab/.venv
ENV UV_LINK_MODE=copy
ENV PATH="/root/gradlab/.venv/bin:${PATH}"
ENV PYTHONPATH=/root/gradlab/src
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV WANDB_DIR=/root/gradlab/runs
ENV WANDB_CACHE_DIR=/root/gradlab/runs/.wandb-cache
ENV WANDB_CONFIG_DIR=/root/gradlab/runs/.wandb-config
ENV WANDB_DATA_DIR=/root/gradlab/runs/.wandb-data

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /root/gradlab

COPY containers/train/gpu-linux-amd64.lock /tmp/gpu-linux-amd64.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "${UV_PROJECT_ENVIRONMENT}" \
    && uv pip install \
        --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
        --require-hashes \
        --no-build \
        --no-deps \
        -r /tmp/gpu-linux-amd64.lock

ARG GPU_PLAN_SHA256=unknown

LABEL org.opencontainers.image.title="gradlab train GPU foundation"
LABEL org.opencontainers.image.description="Locked GPU framework dependencies for gradlab train runtimes"
LABEL org.opencontainers.image.source="https://github.com/tsilva/gradlab"
LABEL io.gradlab.gpu-plan.sha256="${GPU_PLAN_SHA256}"
# gpu-image-inputs-end

# dependency-image-inputs-begin
FROM ${PYTHON_IMAGE} AS dependency-overlay-build

ENV UV_PROJECT_ENVIRONMENT=/opt/gradlab-dependencies
ENV UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /root/gradlab

COPY containers/train/train-dependencies-linux-amd64.lock /tmp/train-dependencies-linux-amd64.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "${UV_PROJECT_ENVIRONMENT}" \
    && uv pip install \
        --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
        --require-hashes \
        --no-deps \
        -r /tmp/train-dependencies-linux-amd64.lock \
    && printf '%s\n' \
        '/root/gradlab/.venv/lib/python3.14/site-packages' \
        > "${UV_PROJECT_ENVIRONMENT}/lib/python3.14/site-packages/gradlab-gpu.pth"

FROM scratch AS dependency-overlay

ARG BUILDKIT_SBOM_SCAN_STAGE=true

COPY --from=dependency-overlay-build /opt/gradlab-dependencies /opt/gradlab-dependencies

FROM ${GPU_BASE} AS dependencies

ARG BUILDKIT_SBOM_SCAN_STAGE=false

ENV UV_PROJECT_ENVIRONMENT=/opt/gradlab-dependencies
ENV PATH="/opt/gradlab-dependencies/bin:/root/gradlab/.venv/bin:${PATH}"

COPY --link --from=dependency-overlay / /

ARG TRAIN_PLAN_SHA256=unknown
ARG GPU_IMAGE=unknown
ARG GPU_IMAGE_DIGEST=unknown

LABEL org.opencontainers.image.title="gradlab train dependencies"
LABEL org.opencontainers.image.description="Locked system and Python dependencies for gradlab train runtimes"
LABEL org.opencontainers.image.source="https://github.com/tsilva/gradlab"
LABEL io.gradlab.train-plan.sha256="${TRAIN_PLAN_SHA256}"
LABEL org.opencontainers.image.base.name="${GPU_IMAGE}"
LABEL org.opencontainers.image.base.digest="${GPU_IMAGE_DIGEST}"

# dependency-image-inputs-end

# runtime-image-inputs-begin
FROM ${PYTHON_IMAGE} AS app-package

ENV UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /src

COPY pyproject.toml ./
COPY THIRD_PARTY_NOTICES.md ./
COPY METRICS.md ./
COPY experiments/goals ./experiments/goals
COPY experiments/recipes ./experiments/recipes
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    printf '# gradlab runtime package\n' > README.md \
    && uv pip install --target /opt/gradlab --no-deps .

FROM scratch AS runtime-overlay

COPY --from=app-package /opt/gradlab /opt/gradlab
COPY METRICS.md /root/gradlab/METRICS.md
COPY src /root/gradlab/src
COPY scripts /root/gradlab/scripts
COPY experiments/goals /root/gradlab/experiments/goals
COPY experiments/recipes /root/gradlab/experiments/recipes
COPY experiments/modal_eval.yaml /root/gradlab/experiments/modal_eval.yaml
COPY --chmod=755 containers/train/gradlab /usr/local/bin/gradlab
COPY --chmod=755 containers/train/entrypoint.sh /usr/local/bin/gradlab-container-entrypoint
COPY --chmod=755 containers/train/smoke.py /usr/local/bin/gradlab-container-smoke

FROM ${RUNTIME_BASE} AS runtime

ENV PYTHONPATH=/root/gradlab/src:/opt/gradlab

COPY --link --from=runtime-overlay / /

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG RUNTIME_INPUT_SHA256=unknown
ARG DEPENDENCY_IMAGE=unknown
ARG DEPENDENCY_IMAGE_DIGEST=unknown

ENV GRADLAB_SOURCE_SHA="${VCS_REF}"
ENV GRADLAB_RUNTIME_INPUT_SHA256="${RUNTIME_INPUT_SHA256}"

LABEL org.opencontainers.image.title="gradlab train runtime"
LABEL org.opencontainers.image.description="Locked gradlab training and evaluation runtime"
LABEL org.opencontainers.image.source="https://github.com/tsilva/gradlab"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL io.gradlab.runtime-input.sha256="${RUNTIME_INPUT_SHA256}"
LABEL org.opencontainers.image.base.name="${DEPENDENCY_IMAGE}"
LABEL org.opencontainers.image.base.digest="${DEPENDENCY_IMAGE_DIGEST}"

CMD ["gradlab-container-smoke"]
# runtime-image-inputs-end
