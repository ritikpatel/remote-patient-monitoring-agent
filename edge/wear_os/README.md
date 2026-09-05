# Wear OS sensor app

PROJECT_PLAN.md section 8, item 6: samples HR + accelerometer on-watch and batches
them over BLE to `edge/edge_agent/`.

## Honest status

**Built and run — on the Wear OS emulator, not a physical watch.** Once Android
Studio's SDK was available, this was verified for real, not just reviewed:

- `sdkmanager` installed a Wear OS 5 (API 34, arm64) system image; `avdmanager`
  created an AVD (`wear_test`); the Gradle wrapper was generated
  (`gradle wrapper --gradle-version 8.10.2`) against this project.
- `./gradlew assembleDebug` builds a debug APK. Two real bugs surfaced and were
  fixed by actually compiling, not by inspection: `android.useAndroidX=true` was
  missing from `gradle.properties`, and `MainActivity` extended plain `Activity`
  instead of `androidx.activity.ComponentActivity` (`registerForActivityResult` only
  exists on the latter). A placeholder adaptive-icon resource was also added — the
  manifest referenced `@mipmap/ic_launcher`, which didn't exist yet.
- Installed and launched on the booted emulator (`adb install` / `adb shell am
  start`). A **third** real bug surfaced only at runtime, not at compile time: the
  app crashed with `SerializationException: Serializer for class 'WatchBatch' is not
  found` — the `kotlinx-serialization-json` runtime library was declared, but the
  `org.jetbrains.kotlin.plugin.serialization` **compiler plugin**, which is what
  actually generates `@Serializable` serializers, was never applied. Fixed in both
  `build.gradle.kts` files.
- After that fix: permissions granted (`adb shell pm grant`), the app launched,
  `SensorGattServerService` started as a foreground service and stayed alive and
  crash-free across multiple `SAMPLE_WINDOW_S` (10 s) batch-emission cycles
  (confirmed via `dumpsys activity services` and a clean `logcat` with zero
  `FATAL EXCEPTION` occurrences after the fix).

**What is still unverified:** BLE GATT central/peripheral pairing end-to-end. The
Wear OS emulator's Bluetooth stack does not reliably support peripheral-mode GATT
advertising, so this run could not confirm that `edge/edge_agent/transport.py`'s
`BleTransport` (real hardware, a real central) actually receives a notification from
this service — only that the service constructs its `BluetoothGattServer` and
`AdvertiseCallback` without throwing. `startAdvertising`'s null-safe call
(`advertiser?.startAdvertising(...)`) means the app degrades safely if the
emulator's virtual adapter has no advertiser at all, rather than crashing. Confirming
the actual notification hand-off needs a physical Wear OS watch and a BLE-capable
central, neither available here.

The `edge/edge_agent/` side of this link is real, tested Python (see its own tests):
`BleTransport` in `edge/edge_agent/transport.py` will connect to this app's GATT
characteristic on real hardware, using the exact UUIDs and JSON batch shape defined
in `edge/edge_agent/protocol.py`.

## What it does

`SensorGattServerService` (a foreground `Service`, so it keeps sampling with the
screen off):

1. Registers `SensorManager` listeners for `TYPE_HEART_RATE` (1 Hz) and
   `TYPE_ACCELEROMETER` (fastest available rate, downsampled to ~5 Hz before
   batching — full-rate accelerometer over BLE would drain the battery for no
   benefit `features.py` actually uses).
2. Buffers samples in memory for `SAMPLE_WINDOW_S` (10 s, `protocol.SAMPLE_WINDOW_S`
   on the Python side — keep the two in sync if you change either).
3. Every window, serializes the batch to the exact JSON shape `protocol.WatchBatch`
   parses, and pushes it as a GATT notification on
   `BATCH_CHARACTERISTIC_UUID` under `SERVICE_UUID` (both hardcoded to match
   `edge/edge_agent/protocol.py` — **do not regenerate these UUIDs on one side
   without updating the other**).
4. Runs a `BluetoothGattServer` (this watch is the BLE *peripheral*; the phone/gateway
   running `edge_agent.py`'s `BleTransport` is the *central* that connects and
   subscribes) so it advertises and accepts one central connection.

## Why GATT peripheral mode, not the Wear OS Data Layer API

Wear OS apps more commonly talk to a *paired phone's own companion Android app* via
Google's Wearable Data Layer API (`MessageClient`/`DataClient`), not raw BLE GATT.
That API needs an Android app on the other end. PROJECT_PLAN.md's `edge/edge_agent/`
is a standalone Python gateway process (any BLE-capable host, not necessarily an
Android phone), which is the GATT central/peripheral model, not the Data Layer model
— so this scaffold implements a `BluetoothGattServer` directly rather than the Data
Layer API. If a future phase instead wants the watch paired to an Android phone app
specifically, that would be a different transport, implemented alongside this one
behind `edge/edge_agent/transport.py`'s `Transport` protocol.

## Building this for real

You would need: Android Studio, the Wear OS SDK platform, a physical Wear OS 3+
watch (or the Wear OS emulator, though `BluetoothGattServer` support in the emulator
is unreliable — test on hardware), and to grant `BODY_SENSORS` and
`BLUETOOTH_ADVERTISE`/`BLUETOOTH_CONNECT` permissions at runtime (Android 12+ runtime
permission prompts, not just the manifest declaration below).

```bash
cd edge/wear_os
./gradlew assembleDebug        # requires Android Studio's SDK on PATH
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
