package `in`.sartaj.classifier

import smile.classification.LogisticRegression
import smile.feature.extraction.BagOfWords
import smile.nlp.tokenizer.SimpleTokenizer

class ActiveClassifier {
    private val models = mutableMapOf<String, LogisticRegression>()
    private var bagOfWords: BagOfWords? = null
    private var isTrained = false

    fun train(documents: List<Document>, labelsMap: Map<String, List<String>>) {
        val allCategories = labelsMap.values.flatten().toSortedSet().toList()
        if (allCategories.isEmpty() || labelsMap.size < 2) {
            isTrained = false
            return
        }

        val corpus = documents.map { "${it.title} ${it.body}" }.toTypedArray()
        
        // FIX: Pass SimpleTokenizer to the BagOfWords constructor
        val tokenizer = SimpleTokenizer(true)
        bagOfWords = BagOfWords(tokenizer, corpus)
        val bow = bagOfWords ?: return

        models.clear()
        for (category in allCategories) {
            val X = mutableListOf<DoubleArray>()
            val y = mutableListOf<Int>()

            for (doc in documents) {
                val tags = labelsMap[doc.path]
                if (tags != null) {
                    val text = "${doc.title} ${doc.body}"
                    
                    // FIX: Convert IntArray word counts to DoubleArray for Logistic Regression
                    val xInt = bow.apply(text)
                    X.add(xInt.map { it.toDouble() }.toDoubleArray())
                    
                    y.add(if (tags.contains(category)) 1 else 0)
                }
            }

            if (y.distinct().size == 2 && X.size >= 2) {
                try {
                    models[category] = LogisticRegression.fit(X.toTypedArray(), y.toIntArray())
                } catch (_: Exception) {}
            }
        }
        isTrained = models.isNotEmpty()
    }

    fun predictProbabilities(text: String): Map<String, Double> {
        if (!isTrained) return emptyMap()
        val bow = bagOfWords ?: return emptyMap()
        
        val xInt = bow.apply(text)
        val xDouble = xInt.map { it.toDouble() }.toDoubleArray()

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

