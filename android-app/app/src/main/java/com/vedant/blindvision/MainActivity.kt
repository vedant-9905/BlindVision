package com.vedant.blindvision

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color as AndroidColor
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview as CameraPreview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt

private val DetectionGreen = Color(0xFF00FF88)
private val HudSurfaceColor = Color(0xCC0A110E)
private val HudBorderColor = Color(0x4000FF88)
private val WarningAmber = Color(0xFFFFB74D)
private val MutedRed = Color(0xFFFF5252)

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(AndroidColor.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(AndroidColor.TRANSPARENT)
        )
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            BlindVisionTheme {
                BlindVisionApp(lifecycleOwner = this)
            }
        }
    }
}

@Composable
private fun BlindVisionTheme(content: @Composable () -> Unit) {
    val colorScheme = darkColorScheme(
        primary = DetectionGreen,
        onPrimary = Color(0xFF031A0F),
        background = Color.Black,
        surface = Color(0xFF101715),
        onBackground = Color.White,
        onSurface = Color.White
    )
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}

@Composable
private fun BlindVisionApp(lifecycleOwner: LifecycleOwner) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    var permissionRequestedOnce by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        permissionRequestedOnce = true
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                hasCameraPermission = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.Black
    ) {
        if (hasCameraPermission) {
            LiveDetectionScreen(lifecycleOwner = lifecycleOwner)
        } else {
            PermissionPromptScreen(
                showSettingsHint = permissionRequestedOnce,
                onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                onOpenSettings = {
                    val intent = Intent(
                        Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.fromParts("package", context.packageName, null)
                    )
                    context.startActivity(intent)
                }
            )
        }
    }
}

@Composable
private fun LiveDetectionScreen(lifecycleOwner: LifecycleOwner) {
    val context = LocalContext.current
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }

    var confidenceThreshold by remember {
        mutableFloatStateOf(ObjectDetectorHelper.DEFAULT_CONFIDENCE_THRESHOLD)
    }
    var isMuted by remember { mutableStateOf(false) }

    val detectorHelper = remember(context) {
        ObjectDetectorHelper(
            context = context,
            confidenceThreshold = ObjectDetectorHelper.DEFAULT_CONFIDENCE_THRESHOLD,
            muted = false
        )
    }

    var telemetry by remember {
        mutableStateOf(FrameTelemetry(modelStatus = detectorHelper.modelStatus))
    }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val analysisExecutor: ExecutorService = remember {
        Executors.newSingleThreadExecutor()
    }

    val previewView = remember(context) {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }
    }

    DisposableEffect(detectorHelper, analysisExecutor) {
        onDispose {
            analysisExecutor.shutdown()
            detectorHelper.close()
        }
    }

    DisposableEffect(lifecycleOwner, previewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        var boundProvider: ProcessCameraProvider? = null
        var isDisposed = false

        cameraProviderFuture.addListener(
            {
                if (isDisposed) return@addListener
                try {
                    val provider = cameraProviderFuture.get()
                    boundProvider = provider

                    val previewUseCase = CameraPreview.Builder()
                        .build()
                        .apply {
                            setSurfaceProvider(previewView.surfaceProvider)
                        }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        try {
                            val result = detectorHelper.detect(imageProxy)
                            if (!isDisposed) {
                                mainExecutor.execute {
                                    telemetry = result
                                }
                            }
                        } finally {
                            imageProxy.close()
                        }
                    }

                    val cameraSelector = when {
                        provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ->
                            CameraSelector.DEFAULT_BACK_CAMERA
                        provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ->
                            CameraSelector.DEFAULT_FRONT_CAMERA
                        else -> throw IllegalStateException("No camera hardware available")
                    }

                    provider.unbindAll()
                    provider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        previewUseCase,
                        imageAnalysis
                    )
                    cameraError = null
                } catch (exc: Exception) {
                    cameraError = exc.localizedMessage ?: "Failed to start camera"
                }
            },
            mainExecutor
        )

        onDispose {
            isDisposed = true
            boundProvider?.unbindAll()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        DetectionOverlayCanvas(
            telemetry = telemetry,
            modifier = Modifier.fillMaxSize()
        )

        TopTelemetryHud(
            telemetry = telemetry,
            cameraError = cameraError,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 12.dp)
        )

        BottomControlCard(
            confidenceThreshold = confidenceThreshold,
            isMuted = isMuted,
            topDetection = telemetry.detections.maxByOrNull { it.confidence },
            onThresholdChange = { updated ->
                confidenceThreshold = updated
                detectorHelper.confidenceThreshold = updated
            },
            onToggleMute = {
                val nextMuted = !isMuted
                isMuted = nextMuted
                detectorHelper.setMuted(nextMuted)
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 16.dp)
        )
    }
}

@Composable
private fun DetectionOverlayCanvas(
    telemetry: FrameTelemetry,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = remember {
        TextStyle(
            color = DetectionGreen,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold
        )
    }

    Canvas(modifier = modifier) {
        if (telemetry.detections.isEmpty()) return@Canvas

        val canvasW = size.width
        val canvasH = size.height
        val srcW = telemetry.sourceWidth
        val srcH = telemetry.sourceHeight

        val drawnW: Float
        val drawnH: Float
        val offsetX: Float
        val offsetY: Float

        if (srcW > 0 && srcH > 0) {
            val scale = max(canvasW / srcW.toFloat(), canvasH / srcH.toFloat())
            drawnW = srcW * scale
            drawnH = srcH * scale
            offsetX = (canvasW - drawnW) * 0.5f
            offsetY = (canvasH - drawnH) * 0.5f
        } else {
            drawnW = canvasW
            drawnH = canvasH
            offsetX = 0f
            offsetY = 0f
        }

        val strokePx = 3.dp.toPx()
        val cornerPx = 8.dp.toPx()
        val badgeCornerPx = 6.dp.toPx()
        val padH = 8.dp.toPx()
        val padV = 4.dp.toPx()
        val badgeGap = 4.dp.toPx()

        clipRect {
            for (detection in telemetry.detections) {
                val box = detection.boundingBox
                val left = box.left * drawnW + offsetX
                val top = box.top * drawnH + offsetY
                val right = box.right * drawnW + offsetX
                val bottom = box.bottom * drawnH + offsetY

                val boxW = right - left
                val boxH = bottom - top
                if (boxW <= 1f || boxH <= 1f) continue

                drawRoundRect(
                    color = DetectionGreen,
                    topLeft = Offset(left, top),
                    size = Size(boxW, boxH),
                    cornerRadius = CornerRadius(cornerPx, cornerPx),
                    style = Stroke(width = strokePx)
                )

                val pct = (detection.confidence * 100f).roundToInt()
                val badgeText = "${detection.label} $pct% · ${detection.proximity} · ${detection.direction}"
                val measured = textMeasurer.measure(text = badgeText, style = labelStyle)

                val badgeW = measured.size.width + padH * 2f
                val badgeH = measured.size.height + padV * 2f
                val maxBadgeX = max(0f, canvasW - badgeW)
                val badgeX = left.coerceIn(0f, maxBadgeX)
                val preferredY = top - badgeH - badgeGap
                val badgeY = if (preferredY >= 0f) preferredY else top.coerceIn(0f, max(0f, canvasH - badgeH))

                drawRoundRect(
                    color = Color(0xD907100C),
                    topLeft = Offset(badgeX, badgeY),
                    size = Size(badgeW, badgeH),
                    cornerRadius = CornerRadius(badgeCornerPx, badgeCornerPx)
                )

                drawText(
                    textLayoutResult = measured,
                    topLeft = Offset(badgeX + padH, badgeY + padV)
                )
            }
        }
    }
}

@Composable
private fun TopTelemetryHud(
    telemetry: FrameTelemetry,
    cameraError: String?,
    modifier: Modifier = Modifier
) {
    val isModelReady = telemetry.modelStatus.contains("Ready", ignoreCase = true)
    val statusText = cameraError ?: telemetry.modelStatus
    val statusAccent = when {
        cameraError != null -> MutedRed
        isModelReady -> DetectionGreen
        else -> WarningAmber
    }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(color = HudSurfaceColor, shape = RoundedCornerShape(16.dp))
            .border(width = 1.dp, color = HudBorderColor, shape = RoundedCornerShape(16.dp))
            .padding(horizontal = 14.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            modifier = Modifier.weight(1f),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(color = statusAccent, shape = CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = statusText,
                color = Color.White,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            MetricPill(
                text = String.format(Locale.US, "%.1f FPS", telemetry.fps),
                accent = DetectionGreen
            )
            MetricPill(
                text = "${telemetry.inferenceTimeMs} ms",
                accent = Color.White
            )
        }
    }
}

@Composable
private fun MetricPill(
    text: String,
    accent: Color
) {
    Box(
        modifier = Modifier
            .background(color = Color(0xFF16241E), shape = RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            color = accent,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun BottomControlCard(
    confidenceThreshold: Float,
    isMuted: Boolean,
    topDetection: DetectionResult?,
    onThresholdChange: (Float) -> Unit,
    onToggleMute: () -> Unit,
    modifier: Modifier = Modifier
) {
    val thresholdPercent = (confidenceThreshold * 100f).roundToInt()
    val muteDescription = if (isMuted) {
        "Audio alerts muted. Double tap to unmute audio."
    } else {
        "Audio alerts active. Double tap to mute audio."
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(color = HudSurfaceColor, shape = RoundedCornerShape(20.dp))
            .border(width = 1.dp, color = HudBorderColor, shape = RoundedCornerShape(20.dp))
            .padding(horizontal = 16.dp, vertical = 14.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Confidence Threshold",
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = if (topDetection != null) {
                        "Lead: ${topDetection.label} ${topDetection.proximity}, ${topDetection.direction}"
                    } else {
                        "Scanning surroundings offline..."
                    },
                    color = Color(0xFFB5C7BF),
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Text(
                text = "$thresholdPercent%",
                color = DetectionGreen,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Slider(
            value = confidenceThreshold,
            onValueChange = onThresholdChange,
            valueRange = 0.30f..0.90f,
            colors = SliderDefaults.colors(
                thumbColor = DetectionGreen,
                activeTrackColor = DetectionGreen,
                inactiveTrackColor = Color(0xFF2B3A33)
            ),
            modifier = Modifier
                .fillMaxWidth()
                .semantics {
                    contentDescription = "Confidence threshold $thresholdPercent percent"
                }
        )

        Spacer(modifier = Modifier.height(6.dp))

        Button(
            onClick = onToggleMute,
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isMuted) Color(0xFF2D1B1B) else DetectionGreen,
                contentColor = if (isMuted) Color.White else Color(0xFF031A0F)
            ),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
                .then(
                    if (isMuted) {
                        Modifier.border(
                            width = 1.5.dp,
                            color = MutedRed,
                            shape = RoundedCornerShape(14.dp)
                        )
                    } else {
                        Modifier
                    }
                )
                .semantics {
                    contentDescription = muteDescription
                }
        ) {
            Image(
                imageVector = if (isMuted) SpeakerOffIcon else SpeakerOnIcon,
                contentDescription = null,
                colorFilter = ColorFilter.tint(
                    if (isMuted) MutedRed else Color(0xFF031A0F)
                ),
                modifier = Modifier.size(22.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = if (isMuted) "Audio Muted — Tap to Unmute" else "Audio Active — Tap to Mute",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun PermissionPromptScreen(
    showSettingsHint: Boolean,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = stringResource(R.string.app_name),
            color = DetectionGreen,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(14.dp))
        Text(
            text = stringResource(
                if (showSettingsHint) {
                    R.string.permission_denied_notice
                } else {
                    R.string.camera_permission_rationale
                }
            ),
            color = Color.White,
            fontSize = 16.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onRequestPermission,
            colors = ButtonDefaults.buttonColors(
                containerColor = DetectionGreen,
                contentColor = Color(0xFF031A0F)
            ),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {
            Text(
                text = stringResource(R.string.grant_camera_permission),
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold
            )
        }
        if (showSettingsHint) {
            Spacer(modifier = Modifier.height(12.dp))
            OutlinedButton(
                onClick = onOpenSettings,
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp)
            ) {
                Text(
                    text = stringResource(R.string.open_app_settings),
                    color = Color.White,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

private val SpeakerOnIcon: ImageVector by lazy {
    ImageVector.Builder(
        name = "SpeakerOn",
        defaultWidth = 24.dp,
        defaultHeight = 24.dp,
        viewportWidth = 24f,
        viewportHeight = 24f
    ).apply {
        path(fill = SolidColor(Color.Black)) {
            moveTo(3f, 9f)
            verticalLineTo(15f)
            horizontalLineTo(7f)
            lineTo(12f, 20f)
            verticalLineTo(4f)
            lineTo(7f, 9f)
            horizontalLineTo(3f)
            close()
            moveTo(16.5f, 12f)
            curveTo(16.5f, 10.23f, 15.48f, 8.71f, 14f, 7.97f)
            verticalLineTo(16.02f)
            curveTo(15.48f, 15.29f, 16.5f, 13.77f, 16.5f, 12f)
            close()
            moveTo(14f, 3.23f)
            verticalLineTo(5.29f)
            curveTo(16.89f, 6.15f, 19f, 8.83f, 19f, 12f)
            curveTo(19f, 15.17f, 16.89f, 17.85f, 14f, 18.71f)
            verticalLineTo(20.77f)
            curveTo(18.01f, 19.86f, 21f, 16.28f, 21f, 12f)
            curveTo(21f, 7.72f, 18.01f, 4.14f, 14f, 3.23f)
            close()
        }
    }.build()
}

private val SpeakerOffIcon: ImageVector by lazy {
    ImageVector.Builder(
        name = "SpeakerOff",
        defaultWidth = 24.dp,
        defaultHeight = 24.dp,
        viewportWidth = 24f,
        viewportHeight = 24f
    ).apply {
        path(fill = SolidColor(Color.Black)) {
            moveTo(16.5f, 12f)
            curveTo(16.5f, 10.23f, 15.48f, 8.71f, 14f, 7.97f)
            verticalLineTo(10.18f)
            lineTo(16.45f, 12.63f)
            curveTo(16.48f, 12.43f, 16.5f, 12.22f, 16.5f, 12f)
            close()
            moveTo(19f, 12f)
            curveTo(19f, 12.94f, 18.8f, 13.82f, 18.46f, 14.64f)
            lineTo(19.97f, 16.15f)
            curveTo(20.63f, 14.91f, 21f, 13.5f, 21f, 12f)
            curveTo(21f, 7.72f, 18.01f, 4.14f, 14f, 3.23f)
            verticalLineTo(5.29f)
            curveTo(16.89f, 6.15f, 19f, 8.83f, 19f, 12f)
            close()
            moveTo(4.27f, 3f)
            lineTo(3f, 4.27f)
            lineTo(7.73f, 9f)
            horizontalLineTo(3f)
            verticalLineTo(15f)
            horizontalLineTo(7f)
            lineTo(12f, 20f)
            verticalLineTo(13.27f)
            lineTo(16.25f, 17.52f)
            curveTo(15.58f, 18.04f, 14.83f, 18.45f, 14f, 18.7f)
            verticalLineTo(20.76f)
            curveTo(15.38f, 20.45f, 16.63f, 19.81f, 17.69f, 18.95f)
            lineTo(19.73f, 21f)
            lineTo(21f, 19.73f)
            lineTo(12f, 10.73f)
            lineTo(4.27f, 3f)
            close()
            moveTo(12f, 4f)
            lineTo(9.91f, 6.09f)
            lineTo(12f, 8.18f)
            verticalLineTo(4f)
            close()
        }
    }.build()
}

@Preview(showBackground = true, backgroundColor = 0xFF000000)
@Composable
private fun ControlsOverlayPreview() {
    BlindVisionTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            TopTelemetryHud(
                telemetry = FrameTelemetry(
                    fps = 24.5f,
                    inferenceTimeMs = 36L,
                    modelStatus = "YOLOv8n FP16 Ready"
                ),
                cameraError = null,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(16.dp)
            )
            BottomControlCard(
                confidenceThreshold = 0.60f,
                isMuted = false,
                topDetection = null,
                onThresholdChange = {},
                onToggleMute = {},
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
            )
        }
    }
}
