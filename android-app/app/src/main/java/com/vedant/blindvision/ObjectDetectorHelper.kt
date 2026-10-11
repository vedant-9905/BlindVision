package com.vedant.blindvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.graphics.RectF
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

/**
 * Handles on-device YOLOv8n FP16 TensorFlow Lite inference, NMS post-processing,
 * spatial zone mapping, and cooldown-gated offline TextToSpeech alerts.
 */
class ObjectDetectorHelper(
    private val context: Context,
    @Volatile var confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    @Volatile private var muted: Boolean = false
) {
    private val interpreterLock = Any()
    private var interpreter: Interpreter? = null
    private var tts: TextToSpeech? = null
    private val labels: List<String>

    @Volatile
    private var isTtsReady: Boolean = false

    @Volatile
    private var isClosed: Boolean = false

    @Volatile
    var modelStatus: String = "Initializing..."
        private set

    private var lastSpokenMs: Long = -SPEECH_COOLDOWN_MS
    private var lastFrameTimestampNs: Long = 0L
    private var smoothedFps: Float = 0f

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS * FLOAT_BYTE_SIZE)
        .order(ByteOrder.nativeOrder())

    private val pixelValues = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val outputChannelFirst = Array(1) { Array(OUTPUT_ROWS) { FloatArray(OUTPUT_BOXES) } }
    private var outputBoxFirst: Array<Array<FloatArray>>? = null
    private var isTransposedOutput: Boolean = false

    private var yuvArgbScratch: IntArray = IntArray(0)
    private var rgbaRowScratch: ByteArray = ByteArray(0)
    private var rgbaPackedScratch: ByteBuffer? = null

    private data class CandidateBox(
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val label: String,
        val confidence: Float
    )

    init {
        labels = loadLabels()
        initializeInterpreter()
        initializeTextToSpeech()
    }

    /**
     * Updates audio mute state and immediately interrupts active speech when muted.
     */
    fun setMuted(isMuted: Boolean) {
        muted = isMuted
        if (isMuted) {
            try {
                tts?.stop()
            } catch (_: Exception) {
            }
        }
    }

    /**
     * Processes a CameraX [ImageProxy], runs YOLOv8n TFLite inference, applies NMS,
     * triggers cooldown-protected TTS announcements, and returns telemetry for UI rendering.
     */
    fun detect(imageProxy: ImageProxy): FrameTelemetry {
        if (isClosed) {
            return FrameTelemetry(modelStatus = modelStatus)
        }

        val activeInterpreter = interpreter
        if (activeInterpreter == null) {
            return FrameTelemetry(modelStatus = modelStatus)
        }

        val startNs = SystemClock.elapsedRealtimeNanos()
        val uprightBitmap = imageProxyToUprightBitmap(imageProxy)
            ?: return FrameTelemetry(modelStatus = modelStatus)

        val sourceWidth = uprightBitmap.width
        val sourceHeight = uprightBitmap.height

        val scaledBitmap = if (sourceWidth == INPUT_SIZE && sourceHeight == INPUT_SIZE) {
            uprightBitmap
        } else {
            Bitmap.createScaledBitmap(uprightBitmap, INPUT_SIZE, INPUT_SIZE, true)
        }

        try {
            fillInputBuffer(scaledBitmap)
        } finally {
            if (scaledBitmap !== uprightBitmap) {
                scaledBitmap.recycle()
            }
            uprightBitmap.recycle()
        }

        val inferenceSucceeded = synchronized(interpreterLock) {
            if (isClosed || interpreter == null) {
                false
            } else {
                if (isTransposedOutput) {
                    val altBuffer = outputBoxFirst
                        ?: Array(1) { Array(OUTPUT_BOXES) { FloatArray(OUTPUT_ROWS) } }
                            .also { outputBoxFirst = it }
                    activeInterpreter.run(inputBuffer, altBuffer)
                } else {
                    activeInterpreter.run(inputBuffer, outputChannelFirst)
                }
                true
            }
        }

        if (!inferenceSucceeded) {
            return FrameTelemetry(modelStatus = modelStatus)
        }

        val threshold = confidenceThreshold.coerceIn(0.05f, 0.99f)
        val detections = parseDetections(threshold)
        maybeSpeakTopDetection(detections)

        val endNs = SystemClock.elapsedRealtimeNanos()
        val latencyMs = max(1L, (endNs - startNs) / 1_000_000L)
        updateSmoothedFps(endNs)

        return FrameTelemetry(
            detections = detections,
            fps = smoothedFps,
            inferenceTimeMs = latencyMs,
            modelStatus = modelStatus,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight
        )
    }

    /**
     * Releases TensorFlow Lite interpreter and Android TextToSpeech resources cleanly.
     */
    fun close() {
        isClosed = true
        synchronized(interpreterLock) {
            try {
                interpreter?.close()
            } catch (_: Exception) {
            } finally {
                interpreter = null
            }
        }
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Exception) {
        } finally {
            tts = null
            isTtsReady = false
        }
    }

    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_FILE_NAME).bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
        } catch (_: IOException) {
            emptyList()
        }
    }

    private fun initializeInterpreter() {
        try {
            val mappedModel = FileUtil.loadMappedFile(context, MODEL_FILE_NAME)
            val options = Interpreter.Options().apply {
                setNumThreads(INTERPRETER_THREADS)
            }
            val tflite = Interpreter(mappedModel, options)
            val outShape = tflite.getOutputTensor(0).shape()
            isTransposedOutput = outShape.size == 3 &&
                outShape[1] == OUTPUT_BOXES &&
                outShape[2] == OUTPUT_ROWS
            interpreter = tflite
            modelStatus = "YOLOv8n FP16 Ready"
        } catch (_: IOException) {
            interpreter = null
            modelStatus = "Model missing: run export_tflite.py"
        } catch (exc: Exception) {
            interpreter = null
            modelStatus = "Model error: ${exc.localizedMessage ?: "initialization failed"}"
        }
    }

    private fun initializeTextToSpeech() {
        tts = TextToSpeech(context.applicationContext) { status ->
            if (isClosed) return@TextToSpeech
            if (status == TextToSpeech.SUCCESS) {
                val langStatus = tts?.setLanguage(Locale.US)
                isTtsReady = langStatus != TextToSpeech.LANG_MISSING_DATA &&
                    langStatus != TextToSpeech.LANG_NOT_SUPPORTED
            } else {
                isTtsReady = false
            }
        }
    }

    private fun imageProxyToUprightBitmap(imageProxy: ImageProxy): Bitmap? {
        val rawBitmap = when (imageProxy.format) {
            PixelFormat.RGBA_8888 -> rgba8888ToBitmap(imageProxy)
            ImageFormat.YUV_420_888 -> yuv420ToBitmap(imageProxy)
            else -> null
        } ?: return null

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) {
            return rawBitmap
        }

        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(
            rawBitmap,
            0,
            0,
            rawBitmap.width,
            rawBitmap.height,
            matrix,
            true
        )
        if (rotatedBitmap !== rawBitmap) {
            rawBitmap.recycle()
        }
        return rotatedBitmap
    }

    private fun rgba8888ToBitmap(imageProxy: ImageProxy): Bitmap? {
        if (imageProxy.planes.isEmpty()) return null
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer.apply { rewind() }
        val width = imageProxy.width
        val height = imageProxy.height
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val packedRowBytes = width * 4

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        if (pixelStride == 4 && rowStride == packedRowBytes) {
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }

        val requiredBytes = width * height * 4
        val packedBuffer = rgbaPackedScratch?.takeIf { it.capacity() >= requiredBytes }
            ?: ByteBuffer.allocateDirect(requiredBytes).also { rgbaPackedScratch = it }
        packedBuffer.clear()

        if (rgbaRowScratch.size < packedRowBytes) {
            rgbaRowScratch = ByteArray(packedRowBytes)
        }

        if (pixelStride == 4) {
            for (row in 0 until height) {
                buffer.position(row * rowStride)
                buffer.get(rgbaRowScratch, 0, packedRowBytes)
                packedBuffer.put(rgbaRowScratch, 0, packedRowBytes)
            }
        } else {
            for (row in 0 until height) {
                val rowBase = row * rowStride
                var outIdx = 0
                for (col in 0 until width) {
                    val pxBase = rowBase + col * pixelStride
                    rgbaRowScratch[outIdx++] = buffer.get(pxBase)
                    rgbaRowScratch[outIdx++] = buffer.get(pxBase + 1)
                    rgbaRowScratch[outIdx++] = buffer.get(pxBase + 2)
                    rgbaRowScratch[outIdx++] = buffer.get(pxBase + 3)
                }
                packedBuffer.put(rgbaRowScratch, 0, packedRowBytes)
            }
        }

        packedBuffer.rewind()
        bitmap.copyPixelsFromBuffer(packedBuffer)
        return bitmap
    }

    private fun yuv420ToBitmap(imageProxy: ImageProxy): Bitmap? {
        if (imageProxy.planes.size < 3) return null
        val width = imageProxy.width
        val height = imageProxy.height
        val totalPixels = width * height
        if (yuvArgbScratch.size < totalPixels) {
            yuvArgbScratch = IntArray(totalPixels)
        }

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        var outIndex = 0
        for (y in 0 until height) {
            val yRowOffset = y * yRowStride
            val uvRow = y shr 1
            val uRowOffset = uvRow * uRowStride
            val vRowOffset = uvRow * vRowStride
            for (x in 0 until width) {
                val uvCol = x shr 1
                val yVal = yBuffer.get(yRowOffset + x * yPixelStride).toInt() and 0xFF
                val uVal = (uBuffer.get(uRowOffset + uvCol * uPixelStride).toInt() and 0xFF) - 128
                val vVal = (vBuffer.get(vRowOffset + uvCol * vPixelStride).toInt() and 0xFF) - 128

                val r = (yVal + ((1436 * vVal) shr 10)).coerceIn(0, 255)
                val g = (yVal - ((352 * uVal + 731 * vVal) shr 10)).coerceIn(0, 255)
                val b = (yVal + ((1815 * uVal) shr 10)).coerceIn(0, 255)
                yuvArgbScratch[outIndex++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(yuvArgbScratch, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun fillInputBuffer(bitmap: Bitmap) {
        bitmap.getPixels(pixelValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        inputBuffer.rewind()
        for (pixel in pixelValues) {
            val r = ((pixel shr 16) and 0xFF) * INV_255
            val g = ((pixel shr 8) and 0xFF) * INV_255
            val b = (pixel and 0xFF) * INV_255
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        inputBuffer.rewind()
    }

    private fun parseDetections(threshold: Float): List<DetectionResult> {
        val candidates = ArrayList<CandidateBox>(32)

        if (!isTransposedOutput) {
            val tensor = outputChannelFirst[0]
            val rowCx = tensor[0]
            val rowCy = tensor[1]
            val rowW = tensor[2]
            val rowH = tensor[3]

            for (anchor in 0 until OUTPUT_BOXES) {
                var bestClassIndex = -1
                var bestScore = -1f
                for (rowIdx in 4 until OUTPUT_ROWS) {
                    val score = tensor[rowIdx][anchor]
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIndex = rowIdx - 4
                    }
                }
                if (bestScore < threshold || bestClassIndex < 0) continue

                candidates.add(
                    CandidateBox(
                        cx = rowCx[anchor],
                        cy = rowCy[anchor],
                        w = rowW[anchor],
                        h = rowH[anchor],
                        label = resolveLabel(bestClassIndex),
                        confidence = bestScore
                    )
                )
            }
        } else {
            val tensor = outputBoxFirst?.get(0) ?: return emptyList()
            for (anchor in 0 until OUTPUT_BOXES) {
                val row = tensor[anchor]
                var bestClassIndex = -1
                var bestScore = -1f
                for (colIdx in 4 until OUTPUT_ROWS) {
                    val score = row[colIdx]
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIndex = colIdx - 4
                    }
                }
                if (bestScore < threshold || bestClassIndex < 0) continue

                candidates.add(
                    CandidateBox(
                        cx = row[0],
                        cy = row[1],
                        w = row[2],
                        h = row[3],
                        label = resolveLabel(bestClassIndex),
                        confidence = bestScore
                    )
                )
            }
        }

        if (candidates.isEmpty()) {
            return emptyList()
        }

        val scaleToNormalized = if (
            candidates.any { it.cx > 2f || it.cy > 2f || it.w > 2f || it.h > 2f }
        ) {
            1f / INPUT_SIZE.toFloat()
        } else {
            1f
        }

        val mappedDetections = ArrayList<DetectionResult>(candidates.size)
        for (candidate in candidates) {
            val cx = candidate.cx * scaleToNormalized
            val cy = candidate.cy * scaleToNormalized
            val w = candidate.w * scaleToNormalized
            val h = candidate.h * scaleToNormalized

            val left = (cx - w * 0.5f).coerceIn(0f, 1f)
            val top = (cy - h * 0.5f).coerceIn(0f, 1f)
            val right = (cx + w * 0.5f).coerceIn(0f, 1f)
            val bottom = (cy + h * 0.5f).coerceIn(0f, 1f)
            if (right <= left || bottom <= top) continue

            val boxCenterX = ((left + right) * 0.5f).coerceIn(0f, 1f)
            val boxHeight = bottom - top

            val direction = when {
                boxCenterX < LEFT_ZONE_THRESHOLD -> "Left"
                boxCenterX > RIGHT_ZONE_THRESHOLD -> "Right"
                else -> "Center"
            }
            val proximity = if (boxHeight > NEAR_HEIGHT_THRESHOLD) "Near" else "Ahead"

            mappedDetections.add(
                DetectionResult(
                    boundingBox = RectF(left, top, right, bottom),
                    label = candidate.label,
                    confidence = candidate.confidence.coerceIn(0f, 1f),
                    direction = direction,
                    proximity = proximity
                )
            )
        }

        return applyNms(mappedDetections)
    }

    private fun applyNms(detections: List<DetectionResult>): List<DetectionResult> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<DetectionResult>(min(sorted.size, MAX_DETECTIONS))

        for (candidate in sorted) {
            var suppressed = false
            for (kept in selected) {
                if (kept.label == candidate.label &&
                    calculateIoU(kept.boundingBox, candidate.boundingBox) > IOU_THRESHOLD
                ) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) {
                selected.add(candidate)
                if (selected.size >= MAX_DETECTIONS) break
            }
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val unionArea = areaA + areaB - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    private fun resolveLabel(classIndex: Int): String {
        val raw = labels.getOrNull(classIndex) ?: return "Object $classIndex"
        return raw.replaceFirstChar { ch ->
            if (ch.isLowerCase()) ch.titlecase(Locale.US) else ch.toString()
        }
    }

    private fun maybeSpeakTopDetection(detections: List<DetectionResult>) {
        if (muted || !isTtsReady || detections.isEmpty()) return
        val engine = tts ?: return
        if (engine.isSpeaking) return

        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastSpokenMs < SPEECH_COOLDOWN_MS) return

        val topDetection = detections.maxByOrNull { it.confidence } ?: return
        val utterance = "${topDetection.label} ${topDetection.proximity}, ${topDetection.direction}"
        val result = engine.speak(
            utterance,
            TextToSpeech.QUEUE_FLUSH,
            null,
            "blindvision_$nowMs"
        )
        if (result == TextToSpeech.SUCCESS) {
            lastSpokenMs = nowMs
        }
    }

    private fun updateSmoothedFps(nowNs: Long) {
        if (lastFrameTimestampNs > 0L) {
            val deltaNs = nowNs - lastFrameTimestampNs
            if (deltaNs > 0L) {
                val instantFps = 1_000_000_000f / deltaNs.toFloat()
                smoothedFps = if (smoothedFps == 0f) {
                    instantFps
                } else {
                    smoothedFps * 0.8f + instantFps * 0.2f
                }
            }
        }
        lastFrameTimestampNs = nowNs
    }

    companion object {
        const val MODEL_FILE_NAME = "yolov8n_float16.tflite"
        const val LABELS_FILE_NAME = "coco_labels.txt"
        const val INPUT_SIZE = 640
        const val INPUT_CHANNELS = 3
        const val FLOAT_BYTE_SIZE = 4
        const val OUTPUT_ROWS = 84
        const val OUTPUT_BOXES = 8400
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.60f
        const val IOU_THRESHOLD = 0.45f
        const val SPEECH_COOLDOWN_MS = 1500L
        const val INTERPRETER_THREADS = 4

        private const val INV_255 = 1.0f / 255.0f
        private const val LEFT_ZONE_THRESHOLD = 0.35f
        private const val RIGHT_ZONE_THRESHOLD = 0.65f
        private const val NEAR_HEIGHT_THRESHOLD = 0.55f
        private const val MAX_DETECTIONS = 20
    }
}
