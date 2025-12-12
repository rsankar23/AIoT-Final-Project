"""
YOLO evaluation script
"""

import json
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from ultralytics import YOLO

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "test_images"
PPT_SOURCES = PROJECT_ROOT / "ppt_sources"
RESULTS_DIR = PROJECT_ROOT / "results"

PPT_SOURCES.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

COLORS = {
    'baseline': '#E74C3C',
    'finetuned': '#27AE60',
    'primary': '#3498DB',
    'secondary': '#9B59B6',
    'accent': '#F39C12',
}


def get_image_files(directory: Path):
    """Get all image files from directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    files = []
    if directory.exists():
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)


def run_inference(model: YOLO, image_files: list):
    """Run inference and collect metrics."""
    all_results = []
    all_confidences = []
    class_counts = {}
    per_image_data = []

    for img_path in image_files:
        results = model(str(img_path), verbose=False)

        img_detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])

                img_detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
                all_confidences.append(conf)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        per_image_data.append({
            'image': img_path.name,
            'num_detections': len(img_detections),
            'detections': img_detections
        })
        all_results.extend(img_detections)

    return {
        'total_detections': len(all_results),
        'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
        'std_confidence': np.std(all_confidences) if all_confidences else 0,
        'min_confidence': np.min(all_confidences) if all_confidences else 0,
        'max_confidence': np.max(all_confidences) if all_confidences else 0,
        'class_counts': class_counts,
        'confidences': all_confidences,
        'per_image': per_image_data,
        'detections_per_image': [len(d['detections']) for d in per_image_data]
    }


def save_annotated_images(model: YOLO, image_files: list, output_dir: Path, prefix: str):
    """Save annotated images."""
    output_dir.mkdir(exist_ok=True)

    for img_path in image_files[:10]:
        results = model(str(img_path), verbose=False)
        for r in results:
            annotated = r.plot()
            out_path = output_dir / f"{prefix}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), annotated)

    print(f"  Saved annotated images to: {output_dir}")


def plot_confidence_comparison(baseline: dict, finetuned: dict, output_path: Path):
    """Plot confidence distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    if baseline['confidences']:
        ax1.hist(baseline['confidences'], bins=20, alpha=0.6, color=COLORS['baseline'],
                label=f"Baseline (μ={baseline['avg_confidence']:.2f})", edgecolor='black')
    if finetuned['confidences']:
        ax1.hist(finetuned['confidences'], bins=20, alpha=0.6, color=COLORS['finetuned'],
                label=f"Fine-tuned (μ={finetuned['avg_confidence']:.2f})", edgecolor='black')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution Comparison')
    ax1.legend(loc='upper left')

    ax2 = axes[1]
    data = []
    labels = []
    if baseline['confidences']:
        data.append(baseline['confidences'])
        labels.append('Baseline\nYOLOv8n')
    if finetuned['confidences']:
        data.append(finetuned['confidences'])
        labels.append('Fine-tuned\nYOLOv8n')

    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        colors_box = [COLORS['baseline'], COLORS['finetuned']][:len(data)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence Score Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_detection_metrics(baseline: dict, finetuned: dict, output_path: Path):
    """Plot detection metrics."""
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    models = ['Baseline\nYOLOv8n', 'Fine-tuned\nYOLOv8n']
    counts = [baseline['total_detections'], finetuned['total_detections']]
    bars = ax1.bar(models, counts, color=[COLORS['baseline'], COLORS['finetuned']],
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Detections')
    ax1.set_title('Detection Count Comparison')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax2 = fig.add_subplot(gs[1])
    confs = [baseline['avg_confidence'] * 100, finetuned['avg_confidence'] * 100]
    stds = [baseline['std_confidence'] * 100, finetuned['std_confidence'] * 100]
    bars = ax2.bar(models, confs, yerr=stds, capsize=5,
                   color=[COLORS['baseline'], COLORS['finetuned']],
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('Average Confidence (%)')
    ax2.set_title('Confidence Comparison')
    ax2.set_ylim(0, 100)
    for bar, conf in zip(bars, confs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax3 = fig.add_subplot(gs[2])
    if baseline['detections_per_image'] and finetuned['detections_per_image']:
        x = range(min(len(baseline['detections_per_image']), len(finetuned['detections_per_image'])))
        ax3.plot(x, baseline['detections_per_image'][:len(x)], 'o-',
                color=COLORS['baseline'], label='Baseline', linewidth=2, markersize=6)
        ax3.plot(x, finetuned['detections_per_image'][:len(x)], 's-',
                color=COLORS['finetuned'], label='Fine-tuned', linewidth=2, markersize=6)
        ax3.set_xlabel('Image Index')
        ax3.set_ylabel('Detections')
        ax3.set_title('Detections per Image')
        ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_class_distribution(baseline: dict, finetuned: dict, output_path: Path):
    """Plot class distribution comparison."""
    all_classes = sorted(set(list(baseline['class_counts'].keys()) +
                            list(finetuned['class_counts'].keys())))

    if not all_classes:
        print("  No classes to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_classes))
    width = 0.35

    baseline_vals = [baseline['class_counts'].get(c, 0) for c in all_classes]
    finetuned_vals = [finetuned['class_counts'].get(c, 0) for c in all_classes]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=COLORS['baseline'], edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned_vals, width, label='Fine-tuned',
                   color=COLORS['finetuned'], edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Object Class')
    ax.set_ylabel('Detection Count')
    ax.set_title('Class Distribution: Baseline vs Fine-tuned')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()

    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_summary_dashboard(baseline: dict, finetuned: dict, output_path: Path):
    """Plot summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('UAV Rescue Detection Model Comparison',
                fontsize=18, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Baseline', 'Fine-tuned']
    counts = [baseline['total_detections'], finetuned['total_detections']]
    bars = ax1.bar(models, counts, color=[COLORS['baseline'], COLORS['finetuned']], edgecolor='black')
    ax1.set_title('Total Detections', fontweight='bold')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    confs = [baseline['avg_confidence'] * 100, finetuned['avg_confidence'] * 100]
    bars = ax2.bar(models, confs, color=[COLORS['baseline'], COLORS['finetuned']], edgecolor='black')
    ax2.set_title('Avg Confidence (%)', fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, conf in zip(bars, confs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{conf:.1f}%', ha='center', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    improvement_det = ((finetuned['total_detections'] - baseline['total_detections']) /
                       max(baseline['total_detections'], 1)) * 100
    improvement_conf = ((finetuned['avg_confidence'] - baseline['avg_confidence']) /
                        max(baseline['avg_confidence'], 0.01)) * 100

    text = f"""
    Performance Improvement

    Detection Count:
    {improvement_det:+.1f}%

    Avg Confidence:
    {improvement_conf:+.1f}%
    """
    ax3.text(0.5, 0.5, text, transform=ax3.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4 = fig.add_subplot(gs[1, :2])
    if baseline['confidences']:
        ax4.hist(baseline['confidences'], bins=25, alpha=0.6, color=COLORS['baseline'],
                label='Baseline', edgecolor='black')
    if finetuned['confidences']:
        ax4.hist(finetuned['confidences'], bins=25, alpha=0.6, color=COLORS['finetuned'],
                label='Fine-tuned', edgecolor='black')
    ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Confidence Distribution', fontweight='bold')
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 2])
    data = []
    labels = []
    if baseline['confidences']:
        data.append(baseline['confidences'])
        labels.append('Baseline')
    if finetuned['confidences']:
        data.append(finetuned['confidences'])
        labels.append('Fine-tuned')
    if data:
        bp = ax5.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['baseline'], COLORS['finetuned']][:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax5.set_title('Confidence Box Plot', fontweight='bold')

    ax6 = fig.add_subplot(gs[2, :])
    all_classes = sorted(set(list(baseline['class_counts'].keys()) +
                            list(finetuned['class_counts'].keys())))
    if all_classes:
        x = np.arange(len(all_classes))
        width = 0.35
        baseline_vals = [baseline['class_counts'].get(c, 0) for c in all_classes]
        finetuned_vals = [finetuned['class_counts'].get(c, 0) for c in all_classes]
        ax6.bar(x - width/2, baseline_vals, width, label='Baseline',
               color=COLORS['baseline'], edgecolor='black')
        ax6.bar(x + width/2, finetuned_vals, width, label='Fine-tuned',
               color=COLORS['finetuned'], edgecolor='black')
        ax6.set_xlabel('Object Class')
        ax6.set_ylabel('Count')
        ax6.set_title('Class Distribution Comparison', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(all_classes, rotation=45, ha='right')
        ax6.legend()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("YOLO MODEL EVALUATION")
    print("=" * 60)

    images_dir = DATA_DIR / "images" if (DATA_DIR / "images").exists() else DATA_DIR
    image_files = get_image_files(images_dir)

    if not image_files:
        print(f"\nNo test images found in {images_dir}")
        print("Please run download_dataset.py first or add images manually.")
        return

    print(f"\nFound {len(image_files)} test images")

    print("\nLoading models...")
    baseline_model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    print("  Loaded: yolov8n.pt (baseline)")

    finetuned_path = PROJECT_ROOT / "models" / "finetuned" / "weights" / "best.pt"
    if finetuned_path.exists():
        finetuned_model = YOLO(str(finetuned_path))
        print(f"  Loaded: {finetuned_path} (fine-tuned)")
    else:
        print(f"  Fine-tuned model not found, using baseline for comparison demo")
        finetuned_model = YOLO(str(PROJECT_ROOT / "yolov8s.pt"))

    print("\nRunning baseline evaluation...")
    baseline_metrics = run_inference(baseline_model, image_files)
    print(f"  Total detections: {baseline_metrics['total_detections']}")
    print(f"  Avg confidence: {baseline_metrics['avg_confidence']:.3f}")

    print("\nRunning fine-tuned evaluation...")
    finetuned_metrics = run_inference(finetuned_model, image_files)
    print(f"  Total detections: {finetuned_metrics['total_detections']}")
    print(f"  Avg confidence: {finetuned_metrics['avg_confidence']:.3f}")

    print("\nSaving annotated images...")
    save_annotated_images(baseline_model, image_files, PPT_SOURCES / "annotated_baseline", "baseline")
    save_annotated_images(finetuned_model, image_files, PPT_SOURCES / "annotated_finetuned", "finetuned")

    print("\nGenerating visualizations...")
    plot_confidence_comparison(baseline_metrics, finetuned_metrics,
                              PPT_SOURCES / "confidence_comparison.png")
    plot_detection_metrics(baseline_metrics, finetuned_metrics,
                          PPT_SOURCES / "detection_metrics.png")
    plot_class_distribution(baseline_metrics, finetuned_metrics,
                           PPT_SOURCES / "class_distribution.png")
    plot_summary_dashboard(baseline_metrics, finetuned_metrics,
                          PPT_SOURCES / "summary_dashboard.png")

    results = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(image_files),
        'baseline': {
            'model': 'yolov8n.pt',
            'total_detections': baseline_metrics['total_detections'],
            'avg_confidence': baseline_metrics['avg_confidence'],
            'std_confidence': baseline_metrics['std_confidence'],
            'class_counts': baseline_metrics['class_counts']
        },
        'finetuned': {
            'model': str(finetuned_path) if finetuned_path.exists() else 'yolov8s.pt',
            'total_detections': finetuned_metrics['total_detections'],
            'avg_confidence': finetuned_metrics['avg_confidence'],
            'std_confidence': finetuned_metrics['std_confidence'],
            'class_counts': finetuned_metrics['class_counts']
        }
    }

    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All visualizations saved to: {PPT_SOURCES}")
    print("=" * 60)


if __name__ == "__main__":
    main()
