package com.nocap.app.diag

import com.nocap.hybrid.HybridPredictor
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * What we persist into CapturedNotification.predictionJson for the diagnostics screen.
 * Decoupled from HybridPredictor.Prediction so the schema can evolve without
 * touching the predictor module.
 */
@Serializable
data class PredictionPayload(
    val source: String,
    val pFinal: Float,
    val pKnn: Float? = null,
    val pHead: Float? = null,
    val neighbours: List<NeighbourSnapshot> = emptyList(),
    val alpha: Float? = null,
) {
    @Serializable
    data class NeighbourSnapshot(
        val id: Long,
        val similarity: Float,
        val label: Float,
        val packageName: String,
    )

    companion object {
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        fun from(prediction: HybridPredictor.Prediction, alpha: Float?): PredictionPayload =
            PredictionPayload(
                source = prediction.source.name,
                pFinal = prediction.importance,
                pKnn = prediction.pKnn,
                pHead = prediction.pHead,
                neighbours = prediction.neighbours.map {
                    NeighbourSnapshot(
                        id = it.id,
                        similarity = it.similarity,
                        label = it.label,
                        packageName = it.packageName,
                    )
                },
                alpha = alpha,
            )

        fun encode(payload: PredictionPayload): String = json.encodeToString(payload)
        fun decode(raw: String): PredictionPayload? = runCatching {
            json.decodeFromString<PredictionPayload>(raw)
        }.getOrNull()
    }
}
