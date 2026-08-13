package `in`.sartaj.classifier

import smile.classification.LogisticRegression
import kotlin.math.ln
import kotlin.math.sqrt

class TfidfVectorizer(private val maxFeatures: Int = 5000) {
    val vocabulary = mutableMapOf<String, Int>()
    val idf = mutableMapOf<String, Double>()

    private fun tokenize(text: String): List<String> {
        return text.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.length > 2 }
    }

    fun fit(corpus: List<String>) {
        val df = mutableMapOf<String, Int>()
        val n = corpus.size

        for (text in corpus) {
            val words = tokenize(text).toSet()
            for (word in words) {
                df[word] = df.getOrDefault(word, 0) + 1
            }
        }

        // FIX 1: Changed it.value > 1 to > 0 so rare keywords aren't deleted.
        // Relaxed upper bound to 0.85 to tolerate slightly more common words.
        val sortedVocab = df.entries
            .filter { it.value > 0 && it.value < n * 0.85 } 
            .sortedByDescending { it.value }
            .take(maxFeatures)

        sortedVocab.forEachIndexed { index, entry ->
            vocabulary[entry.key] = index
            idf[entry.key] = ln((n.toDouble() + 1) / (entry.value.toDouble() + 1)) + 1.0
        }
    }

    fun transform(text: String): DoubleArray {
        val vec = DoubleArray(vocabulary.size)
        if (vocabulary.isEmpty()) return vec

        val words = tokenize(text)
        val tf = mutableMapOf<String, Int>()
        for (word in words) {
            tf[word] = tf.getOrDefault(word, 0) + 1
        }

        var sumSq = 0.0
        for ((word, count) in tf) {
            val idx = vocabulary[word]
            if (idx != null) {
                // FIX 2: Sublinear TF scaling. (1 + ln(tf))
                // Prevents highly repeated words from blinding the AI to other context.
                val sublinearTf = 1.0 + ln(count.toDouble())
                val tfidf = sublinearTf * (idf[word] ?: 1.0)
                vec[idx] = tfidf
                sumSq += tfidf * tfidf
            }
        }

        if (sumSq > 0) {
            val norm = sqrt(sumSq)
            for (i in vec.indices) {
                vec[i] /= norm
            }
        }
        return vec
    }
}

class ActiveClassifier {
    private val models = mutableMapOf<String, LogisticRegression>()
    private var vectorizer: TfidfVectorizer? = null
    private var isTrained = false

    fun train(documents: List<Document>, labelsMap: Map<String, List<String>>) {
        val allCategories = labelsMap.values.flatten().toSortedSet().toList()
        if (allCategories.isEmpty() || labelsMap.size < 2) {
            isTrained = false
            return
        }

        val corpus = documents.map { "${it.title} ${it.body}" }
        val tfidf = TfidfVectorizer(maxFeatures = 8000) // Increased max vocabulary
        tfidf.fit(corpus)
        vectorizer = tfidf

        models.clear()
        for (category in allCategories) {
            val X = mutableListOf<DoubleArray>()
            val y = mutableListOf<Int>()

            for (doc in documents) {
                val tags = labelsMap[doc.path]
                if (tags != null) {
                    X.add(tfidf.transform("${doc.title} ${doc.body}"))
                    y.add(if (tags.contains(category)) 1 else 0)
                }
            }

            val positiveIndices = y.indices.filter { y[it] == 1 }
            val negativeIndices = y.indices.filter { y[it] == 0 }

            // If a tag has only positives or only negatives, skip it safely
            if (positiveIndices.isEmpty() || negativeIndices.isEmpty()) continue

            val balancedX = mutableListOf<DoubleArray>()
            val balancedY = mutableListOf<Int>()

            for (idx in negativeIndices) {
                balancedX.add(X[idx])
                balancedY.add(0)
            }

            for (i in negativeIndices.indices) {
                val idx = positiveIndices[i % positiveIndices.size]
                balancedX.add(X[idx])
                balancedY.add(1)
            }

            if (balancedX.size >= 2) {
                try {
                    models[category] = LogisticRegression.fit(
                        balancedX.toTypedArray(), 
                        balancedY.toIntArray(), 
                        0.001, 1E-4, 500
                    )
                } catch (e: Exception) {
                    println("Failed to train '$category': ${e.message}")
                }
            }
        }
        isTrained = models.isNotEmpty()
    }

    fun predictProbabilities(text: String): Map<String, Double> {
        if (!isTrained) return emptyMap()
        val tfidf = vectorizer ?: return emptyMap()

        val xDouble = tfidf.transform(text)

        val probabilities = mutableMapOf<String, Double>()
        for ((category, model) in models) {
            val posteriors = DoubleArray(2)
            model.predict(xDouble, posteriors)
            probabilities[category] = posteriors[1]
        }
        return probabilities
    }

    fun predict(text: String, threshold: Double = 0.4): List<String> {
        return predictProbabilities(text).filter { it.value > threshold }.keys.toList()
    }
}
