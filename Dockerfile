# ─────────────────────────────────────────────────────────────
# Stage 1: build Marabou from source
#
# Uses python:3.11-bookworm (Debian Bookworm, cmake 3.25).
# cmake 3.28+ removed the FindBoost module Marabou depends on,
# so Bookworm is required.  Python must satisfy Marabou's
# <=3.12,>=3.8 constraint (3.11.x satisfies this cleanly).
# ─────────────────────────────────────────────────────────────
FROM python:3.11-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Marabou and build with Python bindings.
# Boost 1.84 is downloaded and compiled by Marabou's own CMake scripts.
RUN git clone --depth 1 \
    https://github.com/NeuralNetworkVerification/Marabou.git /marabou

WORKDIR /marabou
RUN mkdir build && cd build \
    && cmake .. -DBUILD_PYTHON=ON -DENABLE_OPENBLAS=ON \
    && cmake --build . --parallel "$(nproc)"

# Install maraboupy (editable so compiled .so is resolved at runtime)
RUN pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-bookworm

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /marabou /marabou

# CPU-only PyTorch keeps the image ~1 GB smaller than the CUDA variant
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir onnx onnxruntime numpy

WORKDIR /workspace
COPY . .

CMD ["python", "test.py"]
