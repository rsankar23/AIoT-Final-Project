"""
Flask API for YOLO inference
"""

import os
import sys
import time
import base64
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

app = Flask(__name__)

# Global model
model = None
model_name = None
class_names = None


def load_model(use_finetuned: bool = True):
    """Load YOLO model."""
    global model, model_name, class_names

    finetuned_path = MODELS_DIR / "finetuned" / "weights" / "best.pt"

    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)

    if use_finetuned and finetuned_path.exists():
        model = YOLO(str(finetuned_path))
        model_name = "yolov8n_finetuned"
        print(f"  Loaded fine-tuned model: {finetuned_path}")
    else:
        model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
        model_name = "yolov8n_baseline"
        print("  Loaded baseline model: yolov8n.pt")

    class_names = model.names
    print(f"  Classes: {list(class_names.values())}")
    print("=" * 60)


def process_image(image_data):
    """Process image and run detection."""
    start_time = time.time()

    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    else:
        image = image_data

    if image.mode != 'RGB':
        image = image.convert('RGB')

    results = model(image, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_id": cls_id,
                "class": r.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    latency = time.time() - start_time
    person_detected = any(d['class'].lower() == 'person' for d in detections)

    return {
        "detected": len(detections) > 0,
        "person_detected": person_detected,
        "num_detections": len(detections),
        "detections": detections,
        "latency_ms": round(latency * 1000, 2)
    }


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "UAV Rescue Detection - Final Project",
        "course": "AIoT 4764 - Columbia University",
        "team": "Adib Khondoker, Revath Sankar, Dexin Huang",
        "status": "running",
        "model": model_name,
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /detect": "Run object detection"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "num_classes": len(class_names) if class_names else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })


@app.route("/detect", methods=["POST"])
def detect():
    """Object detection endpoint."""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "Missing 'image' field in JSON"}), 400
            image_data = data["image"]
        elif request.content_type and 'image' in request.content_type:
            image_data = request.data
        else:
            return jsonify({"error": "Invalid content type. Send JSON with base64 image or raw image bytes"}), 400

        print(f"\n{'='*40}")
        print("Detection Request Received")

        result = process_image(image_data)

        print(f"  Detections: {result['num_detections']}")
        print(f"  Person detected: {result['person_detected']}")
        print(f"  Latency: {result['latency_ms']} ms")
        print(f"{'='*40}")

        return jsonify({
            "status": "success",
            **result
        })

    except Exception as e:
        print(f"Error in /detect: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    """Get list of detectable classes."""
    return jsonify({
        "num_classes": len(class_names) if class_names else 0,
        "classes": list(class_names.values()) if class_names else []
    })


def main():
    print("\n" + "=" * 60)
    print("UAV RESCUE DETECTION SERVER")
    print("Columbia University - AIoT 4764 Final Project")
    print("=" * 60 + "\n")

    load_model(use_finetuned=True)
    port = int(os.environ.get("PORT", 8000))

    print(f"\nEndpoints:")
    print(f"  GET  /        - Server info")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /classes - List detectable classes")
    print(f"  POST /detect  - Run detection")
    print(f"\nListening on http://0.0.0.0:{port}")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
