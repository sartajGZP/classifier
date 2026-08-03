package `in`.sartaj.classifier

import com.charleskorn.kaml.Yaml
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class Category(val name: String)

data class Document(val path: String, val title: String, val body: String)

object Storage {
    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }

    fun loadDocuments(): List<Document> {
        val docsDir = File("documents")
        if (!docsDir.exists()) return emptyList()
        return docsDir.walkTopDown().filter { it.extension == "txt" }.sortedBy { it.path }.map { file ->
            val lines = file.readLines()
            val title = lines.firstOrNull() ?: "Untitled"
            val body = if (lines.size > 1) lines.drop(1).joinToString("\n") else ""
            Document(file.path, title, body)
        }.toList()
    }

    fun loadCategories(): MutableList<Category> {
        val file = File("categories.yaml")
        if (!file.exists()) return mutableListOf()
        // The missing import above fixes the type inference here
        val list: List<String> = Yaml.default.decodeFromString(file.readText())
        return list.map { Category(it) }.toMutableList()
    }

    fun saveCategories(categories: List<Category>) {
        val names = categories.map { it.name }
        File("categories.yaml").writeText(Yaml.default.encodeToString(names))
    }

    fun loadLabels(): MutableMap<String, List<String>> {
        val file = File("data/labels.json")
        if (!file.exists()) return mutableMapOf()
        return json.decodeFromString(file.readText())
    }

    fun saveLabels(labels: Map<String, List<String>>) {
        val file = File("data/labels.json")
        file.parentFile?.mkdirs()
        file.writeText(json.encodeToString(labels))
    }
}

