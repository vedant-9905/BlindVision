"""Export YOLOv8n PyTorch weights to FP16 TensorFlow Lite for BlindVision Android."""

from __future__ import annotations

from ultralytics import YOLO

INPUT_IMAGE_SIZE = 640


def export_yolov8n_fp16_tflite() -> None:
    """Download/load YOLOv8n and export to FP16 TensorFlow Lite."""
    print("Loading YOLOv8n model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    print(f"Exporting YOLOv8n to FP16 TensorFlow Lite (imgsz={INPUT_IMAGE_SIZE}, half=True)...")
    exported_output = model.export(
        format="tflite",
        imgsz=INPUT_IMAGE_SIZE,
        half=True,
    )
    print(f"Export complete: {exported_output}")


if __name__ == "__main__":
    export_yolov8n_fp16_tflite()
