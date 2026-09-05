package com.vedant.blindvision

import android.graphics.RectF

/**
 * Immutable representation of a single detected object in normalized screen coordinates.
 *
 * @property boundingBox Normalized bounding box with coordinates clamped to [0f, 1f].
 * @property label Human-readable COCO class label.
 * @property confidence Model confidence score in [0f, 1f].
 * @property direction Horizontal spatial zone ("Left", "Center", or "Right").
 * @property proximity Relative distance estimate ("Near" or "Ahead").
 */
data class DetectionResult(
    val boundingBox: RectF,
    val label: String,
    val confidence: Float,
    val direction: String,
    val proximity: String
)

/**
 * Per-frame inference telemetry and detections emitted to the Compose overlay and HUD.
 *
 * @property detections Post-NMS detections passing the active confidence threshold.
 * @property fps Smoothed processing throughput in frames per second.
 * @property inferenceTimeMs End-to-end detection latency for the latest frame in milliseconds.
 * @property modelStatus Current status of the local TFLite model and interpreter.
 * @property sourceWidth Width of the upright camera analysis frame before 640x640 scaling.
 * @property sourceHeight Height of the upright camera analysis frame before 640x640 scaling.
 */
data class FrameTelemetry(
    val detections: List<DetectionResult> = emptyList(),
    val fps: Float = 0f,
    val inferenceTimeMs: Long = 0L,
    val modelStatus: String = "Initializing...",
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0
)
