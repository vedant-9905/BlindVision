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

        if (cameraError != null) {
            Text(
                text = cameraError.orEmpty(),
                color = WarningAmber,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .statusBarsPadding()
                    .padding(16.dp)
            )
        }
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
