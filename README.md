# BlindVision — Real-Time Object Detection & Audio Assistant

**BlindVision** is a 100% offline, zero-cost assistive Android mobile application built with **Kotlin**, **Jetpack Compose**, **CameraX**, **TensorFlow Lite (YOLOv8n FP16)**, and native Android **Text-to-Speech (TTS)**. It detects nearby objects in real time, estimates their horizontal direction and relative proximity, draws high-contrast visual overlays, and speaks concise spatial alerts without any internet connection, cloud inference, or paid APIs.

---

## Architecture & Data-Flow Pipeline

```text
+------------------------------------------------------------------------------+
|                         CameraX Real-Time Capture                            |
|  PreviewView (FILL_CENTER)  +  ImageAnalysis (RGBA_8888, KEEP_ONLY_LATEST)   |
+--------------------------------------+---------------------------------------+
                                       | Single Background Executor Thread
                                       v
+------------------------------------------------------------------------------+
|                        Pre-Processing (Android Bitmap)                       |
|  Rotate by rotationDegrees -> Resize to 640x640 -> RGB Normalize [0.0, 1.0]  |
|                   Direct ByteBuffer (1 x 640 x 640 x 3 x 4B)                 |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                   On-Device TFLite Inference (4 Threads)                     |
|          yolov8n_float16.tflite  ->  Output Tensor [1, 84, 8400]             |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                       Post-Processing & Spatial Reasoning                    |
|  1. Class Argmax (indices 4..83) & Confidence Filter (default >= 0.60)       |
|  2. Non-Maximum Suppression (IoU <= 0.45) across 8400 candidate boxes        |
|  3. Direction Mapping: Left (<0.35) | Center (0.35..0.65) | Right (>0.65)    |
|  4. Proximity Mapping: Near (box height > 0.55) | Ahead (<= 0.55)            |
+-----------------------+--------------------------------------+---------------+
                        |                                      |
                        v                                      v
+---------------------------------------+  +-----------------------------------+
|      Jetpack Compose Canvas HUD       |  |   Offline Native TextToSpeech     |
|  - High-contrast #00FF88 boxes        |  |  - Locale.US on-device voice      |
|  - FPS, Latency (ms), Model Status    |  |  - 1500ms strict cooldown guard   |
|  - Slider (30%-90%) & Mute Toggle     |  |  - "<Label> <proximity>, <dir>"   |
+---------------------------------------+  +-----------------------------------+
```

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Core Model** | YOLOv8n exported to FP16 TensorFlow Lite (`yolov8n_float16.tflite`) |
| **Input Tensor** | `[1, 640, 640, 3]` Float32 normalized RGB (`[0.0f, 1.0f]`) |
| **Output Tensor** | `[1, 84, 8400]` (4 box coords + 80 COCO class probabilities across 8400 anchors) |
| **Object Classes** | 80 standard MS COCO classes (`coco_labels.txt`) |
| **Performance Target** | 18+ FPS on-device via `STRATEGY_KEEP_ONLY_LATEST` and 4-thread TFLite CPU interpreter |
| **Confidence Threshold** | `0.60f` default (interactive slider `0.30f` – `0.90f`) |
| **NMS IoU Threshold** | `0.45f` |
| **Audio Engine** | Native Android `TextToSpeech` (`Locale.US`), offline-first |
| **Speech Cooldown** | Strict `1500 ms` (`1.5 s`) cooldown + `!tts.isSpeaking` overlap protection |
| **Android SDK** | `minSdk = 26`, `compileSdk = 34`, `targetSdk = 34`, Java 17, Kotlin `1.9.24` |

---

## Repository Structure

```text
BlindVision/
├── README.md
├── .gitignore
├── model-export/
│   ├── requirements.txt
│   └── export_tflite.py
└── android-app/
    ├── build.gradle.kts
    ├── settings.gradle.kts
    ├── gradle.properties
    └── app/
        ├── build.gradle.kts
        ├── proguard-rules.pro
        └── src/
            └── main/
                ├── AndroidManifest.xml
                ├── assets/
                │   └── coco_labels.txt
                ├── res/
                │   └── values/
                │       ├── strings.xml
                │       └── themes.xml
                └── java/com/vedant/blindvision/
                    ├── MainActivity.kt
                    ├── ObjectDetectorHelper.kt
                    └── DetectionResult.kt
```

---

## Setup & Build Instructions

### 1. Export `yolov8n_float16.tflite` into Android Assets

The Android project expects `yolov8n_float16.tflite` inside `android-app/app/src/main/assets/`. If the model file is missing, the app boots safely and displays `"Model missing: run export_tflite.py"` in the top HUD instead of crashing.

**Windows (PowerShell):**

```powershell
cd model-export
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python export_tflite.py
```

**macOS / Linux:**

```bash
cd model-export
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python export_tflite.py
```

`export_tflite.py` automatically downloads `yolov8n.pt` (if absent), exports FP16 TFLite (`imgsz=640, half=True`), and copies `yolov8n_float16.tflite` to `android-app/app/src/main/assets/yolov8n_float16.tflite`.

---

### 2. Build & Run the Android App

#### Option A: Android Studio (Recommended)
1. Open **Android Studio** and select **Open** -> choose the `android-app` folder.
2. Allow Gradle sync to download dependencies (AGP `8.5.2`, Kotlin `1.9.24`, Jetpack Compose BOM `2024.06.00`, CameraX `1.3.4`, TensorFlow Lite `2.16.1`).
3. Connect an Android device (Android 8.0 / API 26+) with USB debugging enabled and click **Run**.

#### Option B: Command Line (`./gradlew` / Gradle)
From inside `android-app/` (if using an existing Gradle wrapper or generating one via `gradle wrapper`):

```bash
cd android-app
./gradlew assembleDebug
```

On Windows PowerShell:

```powershell
cd android-app
.\gradlew.bat assembleDebug
```

To install the debug APK onto a connected device via ADB:

```bash
./gradlew installDebug
```

---

## Usage & Accessibility Controls

1. **Camera Permission:** Grant camera permission when prompted on first launch.
2. **Audio Announcements:** Point the rear camera at surroundings. The highest-confidence detection is spoken aloud once every `1.5` seconds as `"<Label> <proximity>, <direction>"` (for example, `"Person near, Center"` or `"Chair ahead, Left"`).
3. **Confidence Threshold Slider:** Adjust between `30%` and `90%` (`60%` default) at the bottom control card to balance recall vs. precision.
4. **Mute / Unmute Toggle:** High-contrast button with TalkBack `contentDescription` to pause or resume audio announcements instantly.

