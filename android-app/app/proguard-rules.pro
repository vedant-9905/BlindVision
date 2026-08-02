# TensorFlow Lite JNI & reflection protection
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# CameraX classes
-keep class androidx.camera.** { *; }

# Keep annotations and generic signatures
-keepattributes *Annotation*
-keepattributes Signature
