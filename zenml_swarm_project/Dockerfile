FROM python:3.13-slim

WORKDIR /app

# System deps — git is required at runtime by steps/train.py, which shells
# out to `git rev-parse` to tag each model version with the commit it was
# trained from.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project, including .git — needed so `git rev-parse` inside the
# container resolves to a real commit sha instead of 'untracked'.
COPY . .

# Mark /app as a safe git directory (git refuses to operate on repos owned
# by a different user than the process by default, which trips inside
# containers since the image is built as root but files may be bind-mounted
# from a host user).
RUN git config --global --add safe.directory /app

# Install the ZenML MLflow integration (needed for the experiment tracker flavor)
RUN zenml integration install mlflow -y

# Ports: ZenML dashboard (8237) + MLflow UI (5000)
EXPOSE 8237 5000

CMD ["bash"]
