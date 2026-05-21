package com.nocap.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface NotificationDao {
    @Insert
    suspend fun insert(notification: CapturedNotification): Long

    @Query("SELECT * FROM captured_notifications ORDER BY postedAt DESC LIMIT 500")
    fun recent(): Flow<List<CapturedNotification>>

    @Query(
        """
        UPDATE captured_notifications
           SET category = :category,
               importance = :importance,
               summary = :summary,
               reason = :reason,
               classifiedAt = :classifiedAt,
               hidden = :hidden
         WHERE id = :id
        """
    )
    suspend fun applyClassification(
        id: Long,
        category: String,
        importance: Int,
        summary: String,
        reason: String,
        classifiedAt: Long,
        hidden: Boolean,
    )

    @Query("UPDATE captured_notifications SET feedback = :value, feedbackSource = :source WHERE id = :id")
    suspend fun setFeedback(id: Long, value: Int, source: String? = "manual")

    @Query(
        """
        UPDATE captured_notifications
           SET importance = :importance,
               category = :category,
               reason = :reason,
               summary = :summary,
               classifiedAt = :classifiedAt,
               hidden = :hidden,
               predictionJson = :predictionJson
         WHERE id = :id
        """
    )
    suspend fun applyHybridPrediction(
        id: Long,
        importance: Int,
        category: String,
        reason: String,
        summary: String,
        classifiedAt: Long,
        hidden: Boolean,
        predictionJson: String,
    )

    data class PackageStats(
        val packageName: String,
        val total: Int,
        val hiddenCount: Int,
        val avgImportance: Double?,
        val wantedCount: Int,
        val skippedCount: Int,
    )

    @Query(
        """
        SELECT packageName,
               COUNT(*)                                       AS total,
               SUM(CASE WHEN hidden = 1 THEN 1 ELSE 0 END)    AS hiddenCount,
               AVG(importance)                                AS avgImportance,
               SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END)  AS wantedCount,
               SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) AS skippedCount
          FROM captured_notifications
         GROUP BY packageName
         ORDER BY total DESC
        """
    )
    suspend fun packageStats(): List<PackageStats>

    @Query(
        """
        SELECT * FROM captured_notifications
         WHERE reason LIKE 'llm_disagreement%'
         ORDER BY classifiedAt DESC
         LIMIT :limit
        """
    )
    suspend fun recentDisagreements(limit: Int = 50): List<CapturedNotification>

    @Query("SELECT COUNT(*) FROM captured_notifications")
    suspend fun totalCount(): Int

    @Query(
        """
        UPDATE captured_notifications
           SET category = 'failed',
               reason = :reason,
               classifiedAt = :classifiedAt
         WHERE id = :id
        """
    )
    suspend fun markFailed(id: Long, reason: String, classifiedAt: Long)

    @Query(
        """
        UPDATE captured_notifications
           SET category = 'manual',
               importance = :importance,
               reason = 'manual override',
               classifiedAt = :classifiedAt,
               hidden = 0
         WHERE id = :id
        """
    )
    suspend fun applyManualImportance(id: Long, importance: Int, classifiedAt: Long)

    @Query("SELECT COUNT(*) FROM captured_notifications WHERE reason LIKE 'llm_disagreement%' OR reason LIKE 'llm_cold_start%' OR category = 'llm'")
    suspend fun llmCallCount(): Int

    @Query(
        """
        SELECT * FROM captured_notifications
         WHERE feedback != 0 AND packageName = :pkg
         ORDER BY postedAt DESC LIMIT :limit
        """
    )
    suspend fun recentFeedbackForPackage(pkg: String, limit: Int): List<CapturedNotification>

    @Query(
        """
        SELECT * FROM captured_notifications
         WHERE feedback != 0
         ORDER BY postedAt DESC LIMIT :limit
        """
    )
    suspend fun recentFeedback(limit: Int): List<CapturedNotification>

    @Query("SELECT * FROM captured_notifications WHERE notificationKey = :key ORDER BY postedAt DESC LIMIT 1")
    suspend fun findByKey(key: String): CapturedNotification?

    /**
     * Most-recent row for a key that is still live (never removed). Used by the
     * listener to decide whether a new post is an UPDATE of an existing live
     * notification or a fresh notification that reused the key.
     */
    @Query(
        """
        SELECT * FROM captured_notifications
         WHERE notificationKey = :key AND removedAt IS NULL
         ORDER BY postedAt DESC LIMIT 1
        """
    )
    suspend fun findLiveByKey(key: String): CapturedNotification?

    @Query(
        """
        UPDATE captured_notifications
           SET title = :title,
               text = :body,
               postedAt = :postedAt,
               updateCount = updateCount + 1
         WHERE id = :id
        """
    )
    suspend fun recordUpdate(id: Long, title: String, body: String, postedAt: Long)

    @Query("UPDATE captured_notifications SET removedAt = :removedAt WHERE id = :id")
    suspend fun markRemoved(id: Long, removedAt: Long)

    @Query("SELECT * FROM captured_notifications WHERE id = :id LIMIT 1")
    suspend fun findById(id: Long): CapturedNotification?

    @Query(
        """
        SELECT packageName FROM captured_notifications
         GROUP BY packageName
         ORDER BY COUNT(*) DESC
         LIMIT :limit
        """
    )
    suspend fun topPackages(limit: Int): List<String>
}
