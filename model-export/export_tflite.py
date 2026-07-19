"""Export YOLOv8n PyTorch weights to FP16 TensorFlow Lite for BlindVision Android."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

TARGET_MODEL_NAME = "yolov8n_float16.tflite"
INPUT_IMAGE_SIZE = 640


def resolve_exported_tflite(exported_output: object, search_roots: list[Path]) -> Path:
    """Locate the generated FP16 TFLite file from Ultralytics export output."""
    if exported_output is not None:
        candidate = Path(str(exported_output))
        if candidate.is_file() and candidate.name == TARGET_MODEL_NAME:
            return candidate
        if candidate.is_dir():
            nested = candidate / TARGET_MODEL_NAME
            if nested.is_file():
                return nested
            fallback_fp16 = sorted(candidate.rglob("*float16*.tflite"))
            if fallback_fp16:
                return fallback_fp16[0]
        if candidate.is_file() and candidate.suffix == ".tflite":
            return candidate

    matches: list[Path] = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(TARGET_MODEL_NAME))

    if not matches:
        for root in search_roots:
            if root.exists():
                matches.extend(root.rglob("*float16*.tflite"))

    if not matches:
        raise FileNotFoundError(
            f"Unable to locate '{TARGET_MODEL_NAME}' after YOLOv8n TFLite export."
        )

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def export_yolov8n_fp16_tflite() -> Path:
    """Download/load YOLOv8n, export to FP16 TFLite, and copy to Android assets."""
    script_dir = Path(__file__).resolve().parent
    assets_dir = (
        script_dir.parent / "android-app" / "app" / "src" / "main" / "assets"
    )
    destination_path = assets_dir / TARGET_MODEL_NAME

    print("Loading YOLOv8n model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    print(
        f"Exporting YOLOv8n to FP16 TensorFlow Lite (imgsz={INPUT_IMAGE_SIZE}, half=True)..."
    )
    exported_output = model.export(
        format="tflite",
        imgsz=INPUT_IMAGE_SIZE,
        half=True,
    )

    source_tflite = resolve_exported_tflite(
        exported_output=exported_output,
        search_roots=[Path.cwd(), script_dir],
    )

    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_tflite, destination_path)

    file_size_mb = destination_path.stat().st_size / (1024.0 * 1024.0)
    print("Export and deployment completed successfully.")
    print(f"Source TFLite : {source_tflite.resolve()}")
    print(f"Android Asset : {destination_path.resolve()}")
    print(f"Model Size    : {file_size_mb:.2f} MB")
    return destination_path


def main() -> int:
    """CLI entry point for YOLOv8n FP16 TFLite export."""
    try:
        export_yolov8n_fp16_tflite()
        return 0
    except Exception as exc:
        print(f"Error exporting YOLOv8n TFLite model: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

