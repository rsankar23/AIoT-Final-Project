FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY yolov8n.pt .
COPY models/finetuned/weights/best.pt models/finetuned/weights/

EXPOSE 8080

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
