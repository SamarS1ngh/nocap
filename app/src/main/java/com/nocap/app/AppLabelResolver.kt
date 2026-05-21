package com.nocap.app

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.core.graphics.drawable.toBitmap
import java.util.concurrent.ConcurrentHashMap

/**
 * Looks up a notification-source package's user-facing label + icon via PackageManager.
 *
 * Results are cached forever — apps don't rename themselves at runtime. Cache lives
 * for the process lifetime; rebuilt on next launch.
 */
class AppLabelResolver(context: Context) {

    private val appContext = context.applicationContext
    private val cache = ConcurrentHashMap<String, Info>()

    data class Info(val label: String, val icon: Bitmap?)

    fun resolve(packageName: String): Info {
        cache[packageName]?.let { return it }
        val info = computeInfo(packageName)
        cache[packageName] = info
        return info
    }

    private fun computeInfo(packageName: String): Info {
        val pm = appContext.packageManager
        return try {
            val ai = pm.getApplicationInfo(packageName, 0)
            val label = pm.getApplicationLabel(ai).toString()
            val drawable = pm.getApplicationIcon(ai)
            val bitmap = runCatching { drawable.toBitmap(ICON_PX, ICON_PX) }.getOrNull()
            Info(label, bitmap)
        } catch (_: PackageManager.NameNotFoundException) {
            Info(fallbackLabel(packageName), null)
        }
    }

    private fun fallbackLabel(packageName: String): String {
        val last = packageName.substringAfterLast('.', missingDelimiterValue = packageName)
        return last.replaceFirstChar { it.uppercase() }
    }

    companion object {
        // Rasterize once at a modest size; we draw at ~28dp in the header.
        private const val ICON_PX = 96
    }
}
