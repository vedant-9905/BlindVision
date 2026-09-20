package com.vedant.blindvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * Handles CameraX frame preprocessing and YOLOv8n FP16 TensorFlow Lite inference.
 */
class ObjectDetectorHelper(
    private val context: Context,
    @Volatile var confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD
) {
    private val interpreterLock = Any()
    private var interpreter: Interpreter? = null
    private val labels: List<String>

    @Volatile
    private var isClosed: Boolean = false

    @Volatile
    var modelStatus: String = "Initializing..."
        private set

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

    init {
        labels = loadLabels()
        initializeInterpreter()
    }

    fun detect(imageProxy: ImageProxy): FrameTelemetry {
        if (isClosed) {
            return FrameTelemetry(modelStatus = modelStatus)
        }
        val activeInterpreter = interpreter ?: return FrameTelemetry(modelStatus = modelStatus)

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

        val endNs = SystemClock.elapsedRealtimeNanos()
        val latencyMs = max(1L, (endNs - startNs) / 1_000_000L)
        updateSmoothedFps(endNs)

        return FrameTelemetry(
            detections = emptyList(),
            fps = smoothedFps,
            inferenceTimeMs = latencyMs,
            modelStatus = modelStatus,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight
        )
    }

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

    private fun imageProxyToUprightBitmap(imageProxy: ImageProxy): Bitmap? {
        val rawBitmap = when (imageProxy.format) {
            PixelFormat.RGBA_8888 -> rgba8888ToBitmap(imageProxy)
            ImageFormat.YUV_420_888 -> yuv420ToBitmap(imageProxy)
            else -> null
        } ?: return null

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) return rawBitmap

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotatedBitmap = Bitmap.createBitmap(
            rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
        )
        if (rotatedBitmap !== rawBitmap) rawBitmap.recycle()
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
        for (row in 0 until height) {
            buffer.position(row * rowStride)
            buffer.get(rgbaRowScratch, 0, packedRowBytes)
            packedBuffer.put(rgbaRowScratch, 0, packedRowBytes)
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
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) * INV_255)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) * INV_255)
            inputBuffer.putFloat((pixel and 0xFF) * INV_255)
        }
        inputBuffer.rewind()
    }

    private fun updateSmoothedFps(nowNs: Long) {
        if (lastFrameTimestampNs > 0L) {
            val deltaNs = nowNs - lastFrameTimestampNs
            if (deltaNs > 0L) {
                val instantFps = 1_000_000_000f / deltaNs.toFloat()
                smoothedFps = if (smoothedFps == 0f) instantFps else smoothedFps * 0.8f + instantFps * 0.2f
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
        const val INTERPRETER_THREADS = 4
        private const val INV_255 = 1.0f / 255.0f
    }
}
