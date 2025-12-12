# Cloud API - UAV Rescue Detection

REST API for real-time object detection deployed on Google Cloud Run.

## Live Endpoint

**Base URL**: `https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/classes` | GET | List detectable classes |
| `/detect` | POST | Run object detection |

## Detectable Classes

1. Person
2. Dog
3. Backpack
4. Car
5. Bicycle

## Usage

### Health Check
```bash
curl https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/health
```

### Detection (Base64)
```bash
curl -X POST https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 image.jpg)'"}'
```

### Detection (Raw Bytes)
```bash
curl -X POST https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/detect \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

## Response Format

```json
{
  "status": "success",
  "detected": true,
  "person_detected": true,
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class": "Person",
      "confidence": 0.9878,
      "bbox": [120.5, 45.2, 380.1, 520.8]
    }
  ],
  "latency_ms": 145.32
}
```

## Files

| File | Description |
|------|-------------|
| `server.py` | Flask REST API server |
| `train.py` | YOLOv8 fine-tuning script |
| `evaluate.py` | Model evaluation script |
| `download_weights.py` | Download pretrained YOLOv8n weights |
| `Dockerfile` | Container for Cloud Run |
| `cloudbuild.yaml` | GCP Cloud Build config |
| `requirements.txt` | Python dependencies |
| `yolov8n.pt` | Baseline YOLO weights (downloaded) |
| `models/finetuned/weights/best.pt` | Fine-tuned weights (98.1% mAP) |

## Reproducibility

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Weights
```bash
python download_weights.py
```
Downloads YOLOv8n weights from Ultralytics (~6 MB).

### 3. Train Model
```bash
python train.py
```
Fine-tunes YOLOv8n on Open Images V7. Saves to `models/finetuned/weights/best.pt`.

### 4. Run Locally
```bash
python server.py
```
Server runs on `http://localhost:8080`

### 5. Deploy to Cloud Run
```bash
gcloud builds submit --config cloudbuild.yaml
```

## Training Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 98.1% |
| Precision | 97.5% |
| Recall | 95.5% |
| Avg Confidence | 71.2% |

## ESP32-CAM Integration

```cpp
const char* serverUrl = "https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/detect";

// POST JPEG image, receive JSON response
// Check response["person_detected"] for rescue alerts
```

## Architecture

```
ESP32-CAM --> HTTPS POST --> Cloud Run (YOLOv8) --> JSON Response
    |                             |                      |
  Capture                    Inference            person_detected
  Image                      ~150ms                  true/false
```

## Team

- Adib Khondoker
- Revath Sankar
- Dexin Huang

Columbia University - AIoT 4764 Final Project
