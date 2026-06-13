plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.nocap.embedding"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        // Don't compress the TFLite model — MappedByteBuffer needs raw bytes.
        noCompress += "tflite"
    }

    packaging {
        // TFLite ships native libs already inside its AARs.
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    // Flex delegate: the converted MiniLM graph contains TF Select ops (e.g.
    // FlexErf in the GELU activation) that the core runtime can't run. Without
    // this, Interpreter init throws at allocateTensors and the predictor stays
    // null. Keep the version in lockstep with tensorflow-lite above.
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
}
