package com.nocap.vectorstore

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [VectorRow::class], version = 2, exportSchema = false)
abstract class VectorDatabase : RoomDatabase() {
    abstract fun vectors(): VectorDao

    companion object {
        @Volatile
        private var instance: VectorDatabase? = null

        fun get(context: Context): VectorDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    VectorDatabase::class.java,
                    "nocap-vectors.db",
                )
                    // v1 → v2 adds notificationKey + unique index. No user-meaningful
                    // data lives here yet; drop the table on schema change.
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { instance = it }
            }
    }
}
