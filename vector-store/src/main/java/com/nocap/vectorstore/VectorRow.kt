package com.nocap.vectorstore

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "vector_rows",
    indices = [
        Index("postedAt"),
        Index("packageName"),
        Index("notificationKey", unique = true),
    ],
)
data class VectorRow(
    @PrimaryKey(autoGenerate = true) val id: Long = 0L,
    val vectorBytes: ByteArray,
    /** null = unlabeled (predicted, not yet confirmed). 0..1 once labeled. */
    val label: Float?,
    val packageName: String,
    val postedAt: Long,
    /**
     * Source notification's StatusBarNotification.key. Used as the upsert
     * dedup key so re-classifying a notification updates the existing row
     * instead of appending a duplicate. Null only for legacy rows produced
     * before this column existed.
     */
    val notificationKey: String?,
    /** JSON blob of arbitrary metadata. Null when not needed. */
    val metaJson: String?,
) {
    // ByteArray defeats default equals/hashCode. Implement manually so Room/tests behave.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VectorRow) return false
        return id == other.id &&
            vectorBytes.contentEquals(other.vectorBytes) &&
            label == other.label &&
            packageName == other.packageName &&
            postedAt == other.postedAt &&
            notificationKey == other.notificationKey &&
            metaJson == other.metaJson
    }

    override fun hashCode(): Int {
        var r = id.hashCode()
        r = 31 * r + vectorBytes.contentHashCode()
        r = 31 * r + (label?.hashCode() ?: 0)
        r = 31 * r + packageName.hashCode()
        r = 31 * r + postedAt.hashCode()
        r = 31 * r + (notificationKey?.hashCode() ?: 0)
        r = 31 * r + (metaJson?.hashCode() ?: 0)
        return r
    }
}
