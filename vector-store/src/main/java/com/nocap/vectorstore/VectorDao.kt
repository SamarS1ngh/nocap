package com.nocap.vectorstore

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface VectorDao {
    @Insert
    suspend fun insert(row: VectorRow): Long

    /**
     * Pull all rows with non-null labels. kNN scans these only — unlabeled rows
     * are predictions we haven't confirmed yet and shouldn't vote.
     */
    @Query("SELECT * FROM vector_rows WHERE label IS NOT NULL")
    suspend fun allLabeled(): List<VectorRow>

    @Query("SELECT COUNT(*) FROM vector_rows WHERE label IS NOT NULL")
    suspend fun labeledCount(): Int

    @Query("SELECT COUNT(*) FROM vector_rows")
    suspend fun totalCount(): Int

    @Query("UPDATE vector_rows SET label = :label WHERE id = :id")
    suspend fun setLabel(id: Long, label: Float)

    @Query("SELECT id FROM vector_rows WHERE notificationKey = :key LIMIT 1")
    suspend fun findIdByKey(key: String): Long?

    @Query("UPDATE vector_rows SET label = :label WHERE notificationKey = :key")
    suspend fun updateLabelByKey(key: String, label: Float)

    @Query("DELETE FROM vector_rows WHERE postedAt < :cutoffMs")
    suspend fun deleteOlderThan(cutoffMs: Long): Int

    @Query("DELETE FROM vector_rows")
    suspend fun deleteAll()
}
