package com.capstonerpm.wearos

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.bluetooth.*
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.time.Instant
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Foreground service: samples HR + accelerometer, batches every
 * WatchProtocol.SAMPLE_WINDOW_S seconds, and notifies a connected BLE central (the
 * phone/gateway running edge/edge_agent/'s BleTransport) with one JSON WatchBatch
 * per window, on WatchProtocol.BATCH_CHARACTERISTIC_UUID.
 *
 * PROJECT_PLAN.md section 8, item 6. See edge/wear_os/README.md for why this watch
 * is the GATT *peripheral* (advertising, accepting a central's connection) rather
 * than using the Wear OS Data Layer API.
 *
 * UNBUILT / UNTESTED: see edge/wear_os/README.md's "Honest status" section. This
 * has never been compiled -- there is no Android toolchain in the environment that
 * wrote it. Review it as Kotlin/Android API usage, not as verified behaviour.
 */
class SensorGattServerService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var bluetoothGattServer: BluetoothGattServer? = null
    private var subscribedCentral: BluetoothDevice? = null
    private var batchCharacteristic: BluetoothGattCharacteristic? = null

    private val hrSamples = CopyOnWriteArrayList<WatchSample>()
    private val accSamples = CopyOnWriteArrayList<WatchSample>()
    private var windowStart: Instant = Instant.now()
    private val handler = Handler(Looper.getMainLooper())
    private val json = Json { ignoreUnknownKeys = true }

    // Downsample the accelerometer before batching (README: full-rate over BLE is
    // wasted battery -- features.py only needs an RMS magnitude per window).
    private var lastAccEmitMs = 0L
    private val accDownsampleIntervalMs = 200L // ~5 Hz

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        startForeground(NOTIFICATION_ID, buildForegroundNotification())
        registerSensors()
        startGattServer()
        scheduleNextBatch()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // --- Sensors -------------------------------------------------------

    private fun registerSensors() {
        sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) // ~1 Hz
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val offsetMs = java.time.Duration.between(windowStart, Instant.now()).toMillis()
        when (event.sensor.type) {
            Sensor.TYPE_HEART_RATE -> {
                val bpm = event.values.getOrNull(0)?.toDouble() ?: return
                if (bpm > 0) hrSamples.add(WatchSample(offsetMs, "hr", bpm))
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val now = System.currentTimeMillis()
                if (now - lastAccEmitMs < accDownsampleIntervalMs) return
                lastAccEmitMs = now
                // Android reports m/s^2; convert to g to match Empatica-style units
                // used everywhere else in this project (services/contracts/observation.py).
                val g = SensorManager.GRAVITY_EARTH
                accSamples.add(WatchSample(offsetMs, "acc_x", (event.values[0] / g).toDouble()))
                accSamples.add(WatchSample(offsetMs, "acc_y", (event.values[1] / g).toDouble()))
                accSamples.add(WatchSample(offsetMs, "acc_z", (event.values[2] / g).toDouble()))
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- Batching --------------------------------------------------------

    private fun scheduleNextBatch() {
        handler.postDelayed({
            emitBatch()
            scheduleNextBatch()
        }, WatchProtocol.SAMPLE_WINDOW_S * 1000)
    }

    private fun emitBatch() {
        val samples = hrSamples.toList() + accSamples.toList()
        hrSamples.clear()
        accSamples.clear()
        val batch = WatchBatch(
            device_id = Build.MODEL ?: "wear-os-unknown",
            patient_ref = patientRef(),
            battery_pct = batteryPercent(),
            batch_start = windowStart.toString(),
            samples = samples,
        )
        windowStart = Instant.now()
        notifyBatch(json.encodeToString(batch))
    }

    /** Placeholder: a real deployment resolves this from paired-account /
     * provisioning state, not a hardcoded value. */
    private fun patientRef(): String = "Patient/UNSET"

    private fun batteryPercent(): Double? = null // wire up BatteryManager if needed

    // --- GATT peripheral ---------------------------------------------------

    private fun startGattServer() {
        val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothGattServer = bluetoothManager.openGattServer(this, gattServerCallback)

        batchCharacteristic = BluetoothGattCharacteristic(
            WatchProtocol.BATCH_CHARACTERISTIC_UUID,
            BluetoothGattCharacteristic.PROPERTY_NOTIFY or BluetoothGattCharacteristic.PROPERTY_READ,
            BluetoothGattCharacteristic.PERMISSION_READ,
        )
        val service = BluetoothGattService(WatchProtocol.SERVICE_UUID, BluetoothGattService.SERVICE_TYPE_PRIMARY)
        service.addCharacteristic(batchCharacteristic)
        bluetoothGattServer?.addService(service)

        val advertiser = bluetoothManager.adapter?.bluetoothLeAdvertiser
        val settings = AdvertiseSettings.Builder()
            .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
            .setConnectable(true)
            .build()
        val data = AdvertiseData.Builder()
            .addServiceUuid(android.os.ParcelUuid(WatchProtocol.SERVICE_UUID))
            .setIncludeDeviceName(true)
            .build()
        advertiser?.startAdvertising(settings, data, advertiseCallback)
    }

    private fun notifyBatch(payload: String) {
        val central = subscribedCentral ?: return
        val characteristic = batchCharacteristic ?: return
        characteristic.value = payload.toByteArray(Charsets.UTF_8)
        bluetoothGattServer?.notifyCharacteristicChanged(central, characteristic, false)
    }

    private val advertiseCallback = object : AdvertiseCallback() {}

    private val gattServerCallback = object : BluetoothGattServerCallback() {
        override fun onConnectionStateChange(device: BluetoothDevice, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                subscribedCentral = device
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED && device == subscribedCentral) {
                subscribedCentral = null
            }
        }

        override fun onDescriptorWriteRequest(
            device: BluetoothDevice, requestId: Int, descriptor: BluetoothGattDescriptor,
            preparedWrite: Boolean, responseNeeded: Boolean, offset: Int, value: ByteArray,
        ) {
            // The central writes the CCCD descriptor to subscribe to notifications --
            // standard BLE GATT subscribe handshake, no batch-specific logic needed here.
            if (responseNeeded) {
                bluetoothGattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, offset, value)
            }
        }
    }

    // --- Foreground notification --------------------------------------------

    private fun buildForegroundNotification(): Notification {
        val channelId = "sensor_streaming"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "Vitals streaming", NotificationManager.IMPORTANCE_LOW)
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(channel)
        }
        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Capstone RPM")
            .setContentText("Streaming heart rate and movement")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()
    }

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        bluetoothGattServer?.close()
        handler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    companion object {
        private const val NOTIFICATION_ID = 1
    }
}
