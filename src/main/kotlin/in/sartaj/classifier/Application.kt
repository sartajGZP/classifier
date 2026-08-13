package `in`.sartaj.classifier

import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.pebble.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.pebbletemplates.pebble.loader.FileLoader
import java.io.File

val documents = Storage.loadDocuments()
val categories = Storage.loadCategories()
val labels = Storage.loadLabels()
val activeModel = ActiveClassifier()

fun main() {
    activeModel.train(documents, labels)
    embeddedServer(Netty, port = 8080, host = "127.0.0.1", module = Application::module)
        .start(wait = true)
}

fun Application.module() {
    install(Pebble) {
        loader(FileLoader().apply { prefix = "templates" })
    }

    routing {
        get("/") {
            call.respondRedirect("/post/0")
        }

        get("/post/{index}") {
            val index = (call.parameters["index"]?.toIntOrNull() ?: 0)
                .coerceIn(0, (documents.size - 1).coerceAtLeast(0))
            if (documents.isEmpty()) {
                call.respondText("No .txt documents found in /documents")
                return@get
            }

            val doc = documents[index]
            var selected = labels[doc.path] ?: emptyList()
            var isPredicted = false

            val probabilities = activeModel.predictProbabilities("${doc.title} ${doc.body}")
            
            // Ask ML model only if untagged
            if (selected.isEmpty()) {
                selected = probabilities.filter { it.value > 0.4 }.keys.toList()
                isPredicted = selected.isNotEmpty()
            }

            call.respond(PebbleContent("index.html", mapOf(
                "document" to doc,
                "index" to index,
                "total" to documents.size,
                "categories" to categories,
                "selected" to selected,
                "is_predicted" to isPredicted,
                "probabilities" to probabilities
            )))
        }

        post("/post/{index}") {
            val index = (call.parameters["index"]?.toIntOrNull() ?: 0)
            val params = call.receiveParameters()
            val selectedCats = params.getAll("category") ?: emptyList()
            val action = params["action"] ?: "next"
            val docKey = documents[index].path

            if (selectedCats.isNotEmpty()) {
                labels[docKey] = selectedCats
                Storage.saveLabels(labels)

            } else {
                if (labels.containsKey(docKey)) {
                    labels.remove(docKey)
                    Storage.saveLabels(labels)
                }
            }

            val nextIndex = if (action == "previous") {
                (index - 1).coerceAtLeast(0)
            } else {
                (index + 1).coerceAtMost(documents.size - 1)
            }
            call.respondRedirect("/post/$nextIndex")
        }

        post("/add-category") {
            val params = call.receiveParameters()
            val newCat = params["new_category"]?.trim() ?: ""
            val redirectIdx = params["redirect_index"] ?: "0"

            if (newCat.isNotEmpty() && categories.none { it.name.equals(newCat, ignoreCase = true) }) {
                categories.add(Category(newCat))
                Storage.saveCategories(categories)
            }
            call.respondRedirect("/post/$redirectIdx")
        }
	post("/train") {
    activeModel.train(documents, labels)
    call.respondRedirect("/post/0")
}
    }
}
