# Dockerfile - concept-tracer backend
#
# WHY PYTHON 3.12-SLIM (NOT FULL):
# The slim image omits documentation, locale files, and non-essential OS packages,
# reducing the image size significantly. All required system libraries for
# psycopg2-binary and PyMuPDF are present in slim. We only add libgl1 explicitly
# because PyMuPDF's rendering functions require it even for text-only extraction.
#
# WHY NOT ALPINE:
# Alpine uses musl libc instead of glibc. Several of our dependencies (PyMuPDF,
# sentence-transformers/torch) distribute glibc-linked wheels and would require
# compilation from source on Alpine - dramatically increasing build time and
# introducing compatibility risk. Slim is the pragmatic choice here.
#
# Design requirement: Python 3.12+ (design proposal §2, Software requirements)

FROM python:3.12-slim

# System dependencies
# libgl1: required by PyMuPDF for rendering pipeline (present even in text-only mode)
# libglib2.0-0: required by OpenCV/torch transitive dependencies
# curl: useful for health checks and debugging inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first - Docker layer cache means this layer is only
# rebuilt when requirements.txt changes, not on every source code change.
# This is standard Docker best practice for Python projects.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: reduces image size (pip cache not useful in a container layer)
# sentence-transformers pulls torch (~1.5GB) - acceptable for a one-time build
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create output directory (mapped to host volume in docker-compose)
RUN mkdir -p output data

# Default command: show help if no concept is provided
# Override at runtime: docker run ... python main.py "attention mechanism"
CMD ["python", "main.py", "--help"]