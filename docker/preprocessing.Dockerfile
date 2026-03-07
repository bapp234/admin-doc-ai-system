#Base image
FROM python:3.10-slim
# System dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app
# Install Poetry
RUN pip install --no-cache-dir poetry==2.0.0
# Disable virtualenv inside Docker
RUN poetry config virtualenvs.create false
# Copy dependency files first (cache layer)
COPY pyproject.toml poetry.lock ./
# Install dependencies
RUN poetry install --no-interaction --no-ansi --only main
# Copy source code
COPY src ./src
# Environment variables
ENV PYTHONPATH=/app
# Chạy file test mặc định
CMD ["python", "run_pipeline.py"]