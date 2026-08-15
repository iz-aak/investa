# syntax=docker/dockerfile:1.7
#
# Multi-stage build for Investa.
#
# Stage 1 ("builder") compiles the TA-Lib C library and Python dependencies
# into a virtualenv. Stage 2 ("runtime") copies just the venv and application
# code into a slim final image — no compilers, no build headers.

# ─── Stage 1: builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build TA-Lib C library from source (required by the ta-lib Python wrapper).
ARG TA_LIB_VERSION=0.4.0
RUN wget -q "http://prdownloads.sourceforge.net/ta-lib/ta-lib-${TA_LIB_VERSION}-src.tar.gz" -O /tmp/ta-lib.tgz \
    && tar -xzf /tmp/ta-lib.tgz -C /tmp \
    && cd /tmp/ta-lib \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/ta-lib /tmp/ta-lib.tgz

# Install Python deps into an isolated venv.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# ─── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# TA-Lib C runtime library (no compilers needed in the final image).
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY investa /app/investa
COPY data/universes /app/data/universes

# Runtime data directory (mount a volume here in compose / prod).
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000

# Default command runs the API. The compose file overrides this for the worker.
CMD ["uvicorn", "investa.main:app", "--host", "0.0.0.0", "--port", "8000"]
