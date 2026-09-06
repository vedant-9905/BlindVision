package com.vedant.blindvision

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException

/**
 * Loads the on-device YOLOv8n FP16 TensorFlow Lite model and COCO class labels.
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

    init {
        labels = loadLabels()
        initializeInterpreter()
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
            interpreter = Interpreter(mappedModel, options)
            modelStatus = "YOLOv8n FP16 Ready"
        } catch (_: IOException) {
            interpreter = null
            modelStatus = "Model missing: run export_tflite.py"
        } catch (exc: Exception) {
            interpreter = null
            modelStatus = "Model error: ${exc.localizedMessage ?: "initialization failed"}"
        }
    }

    companion object {
        const val MODEL_FILE_NAME = "yolov8n_float16.tflite"
        const val LABELS_FILE_NAME = "coco_labels.txt"
        const val INPUT_SIZE = 640
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.60f
        const val INTERPRETER_THREADS = 4
    }
}
