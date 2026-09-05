# Wear OS sensor app

PROJECT_PLAN.md section 8, item 6: samples HR + accelerometer on-watch and batches
them over BLE to `edge/edge_agent/`.

## Honest status

**This has not been built or run.** This development environment has no Android SDK,
no Gradle, no Android/Wear OS emulator, and no physical watch — there is no toolchain
here capable of compiling or executing it. What follows is a structurally complete
Kotlin/Gradle scaffold: real Android APIs used correctly, matching exactly the wire
contract `edge/edge_agent/protocol.py` already implements and is tested against — but
it is unverified beyond compiling correctly. Treat it as a reviewed starting point for
whoever builds this in Android Studio, not as tested code.

The `edge/edge_agent/` side of this link is real, tested Python (see its own tests):
`BleTransport` in `edge/edge_agent/transport.py` will connect to this app's GATT
characteristic once it exists on real hardware, using the exact UUIDs and JSON batch
shape defined in `edge/edge_agent/protocol.py`.

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
