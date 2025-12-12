"""
Fine-tune YOLOv8 on Open Images V7.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "train"
MODELS_DIR = PROJECT_ROOT / "models"
PPT_SOURCES = PROJECT_ROOT / "ppt_sources"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
PPT_SOURCES.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'patience': 10,
    'save': True,
    'plots': True,
}


def plot_training_curves(results_dir: Path, output_path: Path):
    """Plot training curves."""
    csv_path = results_dir / "results.csv"

    if not csv_path.exists():
        print(f"  No results.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Train', linewidth=2)
    if 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/box_loss'], 'r--', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Box Loss')
    ax1.set_title('Box Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if 'train/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Class Loss')
    ax2.set_title('Classification Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', linewidth=2, marker='o', markersize=4)
        ax3.fill_between(df['epoch'], df['metrics/mAP50(B)'], alpha=0.3, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP@0.5')
    ax3.set_title('mAP@0.5', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if 'metrics/precision(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/precision(B)'], 'b-', label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/recall(B)'], 'orange', label='Recall', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision & Recall', fontweight='bold')
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('YOLOv8 Fine-tuning Training Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def copy_yolo_plots(results_dir: Path, output_dir: Path):
    """Copy YOLO plots to ppt_sources."""
    plot_files = [
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'F1_curve.png',
        'P_curve.png',
        'R_curve.png',
        'PR_curve.png',
        'results.png',
    ]

    for plot_file in plot_files:
        src = results_dir / plot_file
        if src.exists():
            dst = output_dir / f"training_{plot_file}"
            shutil.copy(src, dst)
            print(f"  Copied: {plot_file}")


def main():
    print("=" * 60)
    print("YOLOV8 FINE-TUNING")
    print("=" * 60)

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"\nDataset not found at {data_yaml}")
        print("Please run download_dataset.py first.")
        return

    print(f"\nDataset: {data_yaml}")
    print(f"Config: {CONFIG}")

    print("\nLoading base model...")
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

    print("\nStarting training...")
    print("-" * 40)

    results = model.train(
        data=str(data_yaml),
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],
        patience=CONFIG['patience'],
        save=CONFIG['save'],
        plots=CONFIG['plots'],
        project=str(MODELS_DIR),
        name="finetuned",
        exist_ok=True,
    )

    print("-" * 40)
    print("Training complete")

    results_dir = MODELS_DIR / "finetuned"

    print("\nGenerating visualizations...")
    plot_training_curves(results_dir, PPT_SOURCES / "training_curves.png")

    print("\nCopying plots...")
    copy_yolo_plots(results_dir, PPT_SOURCES)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'data_yaml': str(data_yaml),
        'best_model': str(results_dir / "weights" / "best.pt"),
        'final_metrics': {
            'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        }
    }

    summary_file = RESULTS_DIR / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved: {summary_file}")
    print(f"Best model: {results_dir / 'weights' / 'best.pt'}")
    print(f"Plots saved to: {PPT_SOURCES}")

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
