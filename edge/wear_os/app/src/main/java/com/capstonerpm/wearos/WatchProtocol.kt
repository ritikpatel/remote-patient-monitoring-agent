package com.capstonerpm.wearos

import kotlinx.serialization.Serializable
import java.util.UUID

/**
 * Mirrors edge/edge_agent/protocol.py exactly -- both sides of the BLE link parse
 * this same JSON shape. Keep SAMPLE_WINDOW_S, SERVICE_UUID and
 * BATCH_CHARACTERISTIC_UUID in sync with the Python module if you ever change them.
 */
object WatchProtocol {
    val SERVICE_UUID: UUID = UUID.fromString("7a2e1400-9c3b-4b8a-9e2a-6f5c2d9a1001")
    val BATCH_CHARACTERISTIC_UUID: UUID = UUID.fromString("7a2e1401-9c3b-4b8a-9e2a-6f5c2d9a1001")
    const val SAMPLE_WINDOW_S: Long = 10L
}

@Serializable
data class WatchSample(
    val offset_ms: Long,
    val channel: String, // "hr" | "acc_x" | "acc_y" | "acc_z"
    val value: Double,
)

@Serializable
data class WatchBatch(
    val device_id: String,
    val patient_ref: String,
    val battery_pct: Double?,
    val batch_start: String, // ISO-8601, e.g. java.time.Instant.toString()
    val samples: List<WatchSample>,
)
