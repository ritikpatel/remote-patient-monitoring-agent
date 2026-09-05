// Root build file. See app/build.gradle.kts for the actual app module.
plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    // Required for @Serializable to actually generate a serializer at compile time --
    // the kotlinx-serialization-json runtime dependency alone is not enough (found by
    // running the app: SerializationException: Serializer for class 'WatchBatch' is
    // not found).
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.24" apply false
}
