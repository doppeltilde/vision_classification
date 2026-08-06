FROM python:3.14.7-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip list | grep -E 'torch|nvidia|triton|onnxruntime' && echo "Installation verified"

COPY . /app

CMD ["fastapi", "run", "main.py", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]