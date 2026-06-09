# Hugging Face Spaces Docker image for the Singing Evaluation demo.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin matching torch + torchaudio versions (prevents undefined symbol errors)
ARG TORCH_VERSION=2.3.1
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Re-pin torch after other packages (demucs/torchcrepe can pull mismatched versions)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
