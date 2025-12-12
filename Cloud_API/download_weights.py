"""
Download YOLOv8 weights
"""

from ultralytics import YOLO
import os

def download_weights():
    """Download YOLOv8n weights from Ultralytics."""

    print("=" * 50)
    print("Downloading YOLOv8n Weights")
    print("=" * 50)

    print("\nDownloading yolov8n.pt from Ultralytics...")
    model = YOLO("yolov8n.pt")

    if os.path.exists("yolov8n.pt"):
        size_mb = os.path.getsize("yolov8n.pt") / (1024 * 1024)
        print(f"Downloaded: yolov8n.pt ({size_mb:.1f} MB)")

    print("\nModel info:")
    print(f"  Model: YOLOv8n (nano)")
    print(f"  Parameters: ~3.2M")
    print(f"  Trained on: COCO (80 classes)")

    print("\n" + "=" * 50)
    print("Download complete")
    print("=" * 50)
    print("\nNext:")
    print("  1. python train.py")
    print("  2. python server.py")

    return model

if __name__ == "__main__":
    download_weights()
