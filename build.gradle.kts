plugins {
    kotlin("jvm") version "1.9.23"
    kotlin("plugin.serialization") version "1.9.23"
    application
}

application {
    mainClass.set("in.sartaj.classifier.ApplicationKt")
}

repositories {
    mavenCentral()
}

dependencies {
    // Web Server & Pebble Templates (Jinja2 equivalent)
    implementation("io.ktor:ktor-server-core-jvm:2.3.9")
    implementation("io.ktor:ktor-server-netty-jvm:2.3.9")
    implementation("io.ktor:ktor-server-pebble-jvm:2.3.9")
    implementation("io.ktor:ktor-server-content-negotiation-jvm:2.3.9")
    implementation("io.ktor:ktor-serialization-kotlinx-json-jvm:2.3.9")

    // Serialization (JSON + YAML)
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("com.charleskorn.kaml:kaml:0.58.0") // YAML support

    // Machine Learning (Smile ML - No C++ dependencies!)
    implementation("com.github.haifengl:smile-nlp:3.0.2")
    implementation("com.github.haifengl:smile-kotlin:3.0.2")
}

