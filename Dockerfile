FROM python:3.12.11-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["fastapi", "run", "main.py", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
