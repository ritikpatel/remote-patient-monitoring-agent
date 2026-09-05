package com.capstonerpm.wearos

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Requests the runtime permissions SensorGattServerService needs, then starts it.
 * No UI beyond that -- this app has no reason to be opened after setup; it should
 * just keep the foreground service alive.
 */
class MainActivity : Activity() {

    private val requiredPermissions = arrayOf(
        Manifest.permission.BODY_SENSORS,
        Manifest.permission.BLUETOOTH_ADVERTISE,
        Manifest.permission.BLUETOOTH_CONNECT,
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { results ->
        if (results.values.all { it }) {
            startForegroundService(Intent(this, SensorGattServerService::class.java))
        }
        finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val missing = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            startForegroundService(Intent(this, SensorGattServerService::class.java))
            finish()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }
}
