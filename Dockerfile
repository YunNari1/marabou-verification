# ─────────────────────────────────────────────────────────────
# Stage 1: build Marabou (C++ compilation takes ~5-10 minutes)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and build Marabou (installs maraboupy into the Python path)
RUN pip install --no-cache-dir \
    "maraboupy @ https://github.com/NeuralNetworkVerification/Marabou/archive/refs/heads/master.zip"

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Copy the compiled Marabou packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install CPU-only PyTorch (keeps the image ~1 GB smaller than the CUDA build)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime \
    numpy

WORKDIR /workspace

# Copy project files into the image
COPY . .

# Default command: run the verification script
CMD ["python", "test.py"]
