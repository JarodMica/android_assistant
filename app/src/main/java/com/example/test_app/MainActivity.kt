package com.example.test_app

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.AudioModelOptions
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.compose.runtime.derivedStateOf

// -------------------- Application --------------------

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        appContext = applicationContext
    }

    companion object {
        lateinit var appContext: Context
    }
}

enum class Screen {
    Chat,
    Video
}

enum class SettingsTab {
    LLM,
    TTS
}

// Settings data

data class LlmSettingsConfig(
    val systemPrompt: String,
    val temperature: Float,
    val topP: Float,
    val topK: Int
)

data class TtsSettingsConfig(
    val apiKey: String,
    val voiceId: String,
    val model: String,
    val locale: String,
    val style: String,
    val sampleRate: Int
)

// -------------------- Main Activity --------------------

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val appContext = applicationContext

        // Load persisted TTS settings
        val prefs = appContext.getSharedPreferences("murf_prefs", Context.MODE_PRIVATE)
        val initialTtsApiKey = prefs.getString("tts_api_key", "") ?: ""
        val initialTtsVoiceId = prefs.getString("tts_voice_id", "Miles") ?: "Miles"
        val initialTtsModel = prefs.getString("tts_model", "FALCON") ?: "FALCON"
        val initialTtsLocale = prefs.getString("tts_locale", "en-UK") ?: "en-UK"
        val initialTtsStyle = prefs.getString("tts_style", "Calm") ?: "Calm"
        val initialTtsSampleRate = prefs.getInt("tts_sample_rate", 48000)

        setContent {
            val inferenceManager = remember { LlmInferenceManager(appContext) }
            val ttsClient = remember {
                MurfTtsClient(
                    context = appContext,
                    apiKey = initialTtsApiKey
                ).also {
                    it.updateConfig(
                        TtsSettingsConfig(
                            apiKey = initialTtsApiKey,
                            voiceId = initialTtsVoiceId,
                            model = initialTtsModel,
                            locale = initialTtsLocale,
                            style = initialTtsStyle,
                            sampleRate = initialTtsSampleRate
                        )
                    )
                }
            }

            val initialLlmSystemPrompt = prefs.getString(
                "llm_system_prompt",
                "You are a friendly assistant, talking with a user. Respond without Emojis or Symbols."
            ) ?: "You are a friendly assistant, talking with a user. Respond without Emojis or Symbols."

            val initialLlmTemperature = prefs.getFloat("llm_temperature", 0.8f)
            val initialLlmTopK = prefs.getInt("llm_top_k", 40)

            // Shared chat + system logs across both screens
            val sharedMessages = remember { mutableStateListOf<Message>() }
            val sharedSystemMessages = remember { mutableStateListOf<String>() }

            var currentScreen by remember { mutableStateOf(Screen.Chat) }

            // Drawer + settings state
            val drawerState = rememberDrawerState(DrawerValue.Closed)
            val drawerScope = rememberCoroutineScope()
            var currentSettingsTab by remember { mutableStateOf(SettingsTab.LLM) }

            // LLM settings UI state (with default system prompt)
            var llmSystemPrompt by remember {
                mutableStateOf(initialLlmSystemPrompt)
            }
            var llmTempText by remember {
                mutableStateOf(initialLlmTemperature.toString())
            }
            var llmTopKText by remember {
                mutableStateOf(initialLlmTopK.toString())
            }

            // TTS settings UI state (initialized from persisted settings)
            var ttsApiKeyText by remember { mutableStateOf(initialTtsApiKey) }
            var ttsVoiceIdText by remember { mutableStateOf(initialTtsVoiceId) }
            var ttsModelText by remember { mutableStateOf(initialTtsModel) }
            var ttsLocaleText by remember { mutableStateOf(initialTtsLocale) }
            var ttsStyleText by remember { mutableStateOf(initialTtsStyle) }
            var ttsSampleRateText by remember { mutableStateOf(initialTtsSampleRate.toString()) }

            // Automatically push LLM settings into the manager whenever fields change
            LaunchedEffect(llmSystemPrompt, llmTempText, llmTopKText) {
                val temp = llmTempText.toFloatOrNull() ?: 0.8f
                val topK = llmTopKText.toIntOrNull() ?: 40
                val config = LlmSettingsConfig(
                    systemPrompt = llmSystemPrompt,
                    temperature = temp,
                    topP = 0.95f,
                    topK = topK
                )
                inferenceManager.updateSettings(config)

                prefs.edit()
                    .putString("llm_system_prompt", llmSystemPrompt)
                    .putFloat("llm_temperature", temp)
                    .putInt("llm_top_k", topK)
                    .apply()
            }

            fun applyTtsSettings() {
                val sr = ttsSampleRateText.toIntOrNull() ?: 48000
                val config = TtsSettingsConfig(
                    apiKey = ttsApiKeyText,
                    voiceId = ttsVoiceIdText,
                    model = ttsModelText,
                    locale = ttsLocaleText,
                    style = ttsStyleText,
                    sampleRate = sr
                )
                ttsClient.updateConfig(config)

                // Persist to SharedPreferences
                prefs.edit()
                    .putString("tts_api_key", ttsApiKeyText)
                    .putString("tts_voice_id", ttsVoiceIdText)
                    .putString("tts_model", ttsModelText)
                    .putString("tts_locale", ttsLocaleText)
                    .putString("tts_style", ttsStyleText)
                    .putInt("tts_sample_rate", sr)
                    .apply()
            }

            MaterialTheme {
                ModalNavigationDrawer(
                    drawerState = drawerState,
                    drawerContent = {
                        ModalDrawerSheet {
                            Column(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                Text("Settings", style = MaterialTheme.typography.titleMedium)

                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    FilterChip(
                                        selected = currentSettingsTab == SettingsTab.LLM,
                                        onClick = { currentSettingsTab = SettingsTab.LLM },
                                        label = { Text("LLM Settings") }
                                    )
                                    FilterChip(
                                        selected = currentSettingsTab == SettingsTab.TTS,
                                        onClick = { currentSettingsTab = SettingsTab.TTS },
                                        label = { Text("TTS Settings") }
                                    )
                                }

                                when (currentSettingsTab) {
                                    SettingsTab.LLM -> {
                                        Column(
                                            verticalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            OutlinedTextField(
                                                value = llmSystemPrompt,
                                                onValueChange = { llmSystemPrompt = it },
                                                label = { Text("System Prompt") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = false,
                                                maxLines = 4
                                            )
                                            OutlinedTextField(
                                                value = llmTempText,
                                                onValueChange = { llmTempText = it },
                                                label = { Text("Temperature") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            OutlinedTextField(
                                                value = llmTopKText,
                                                onValueChange = { llmTopKText = it },
                                                label = { Text("Top-K") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                        }
                                    }

                                    SettingsTab.TTS -> {
                                        Column(
                                            verticalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            var apiKeyVisible by remember { mutableStateOf(false) }

                                            OutlinedTextField(
                                                value = ttsApiKeyText,
                                                onValueChange = { ttsApiKeyText = it },
                                                label = { Text("Murf API Key") },
                                                placeholder = { Text("Enter your API key") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true,
                                                visualTransformation = if (apiKeyVisible) {
                                                    VisualTransformation.None
                                                } else {
                                                    PasswordVisualTransformation()
                                                },
                                                trailingIcon = {
                                                    val icon =
                                                        if (apiKeyVisible) Icons.Filled.VisibilityOff else Icons.Filled.Visibility
                                                    val description =
                                                        if (apiKeyVisible) "Hide API key" else "Show API key"

                                                    IconButton(onClick = { apiKeyVisible = !apiKeyVisible }) {
                                                        Icon(
                                                            imageVector = icon,
                                                            contentDescription = description
                                                        )
                                                    }
                                                }
                                            )

                                            OutlinedTextField(
                                                value = ttsVoiceIdText,
                                                onValueChange = { ttsVoiceIdText = it },
                                                label = { Text("Voice ID") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            OutlinedTextField(
                                                value = ttsModelText,
                                                onValueChange = { ttsModelText = it },
                                                label = { Text("Model") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            OutlinedTextField(
                                                value = ttsLocaleText,
                                                onValueChange = { ttsLocaleText = it },
                                                label = { Text("Locale (multiNativeLocale)") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            OutlinedTextField(
                                                value = ttsStyleText,
                                                onValueChange = { ttsStyleText = it },
                                                label = { Text("Style") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            OutlinedTextField(
                                                value = ttsSampleRateText,
                                                onValueChange = { ttsSampleRateText = it },
                                                label = { Text("Sample Rate") },
                                                modifier = Modifier.fillMaxWidth(),
                                                singleLine = true
                                            )
                                            Spacer(modifier = Modifier.height(8.dp))
                                            Button(
                                                onClick = {
                                                    applyTtsSettings()
                                                    drawerScope.launch { drawerState.close() }
                                                }
                                            ) {
                                                Text("Apply TTS Settings")
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ) {
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        when (currentScreen) {
                            Screen.Chat -> SimpleChatUI(
                                context = appContext,
                                inferenceManager = inferenceManager,
                                ttsClient = ttsClient,
                                messages = sharedMessages,
                                systemMessages = sharedSystemMessages,
                                onNavigateToVideo = { currentScreen = Screen.Video },
                                onOpenSettings = {
                                    drawerScope.launch { drawerState.open() }
                                }
                            )

                            Screen.Video -> VideoChatUI(
                                context = appContext,
                                inferenceManager = inferenceManager,
                                ttsClient = ttsClient,
                                messages = sharedMessages,
                                systemMessages = sharedSystemMessages,
                                onBack = { currentScreen = Screen.Chat }
                            )
                        }
                    }
                }
            }
        }
    }
}

// -------------------- Data Models --------------------

data class Message(
    val text: String,
    val isUser: Boolean,
    val isTyping: Boolean = false,
    val image: Bitmap? = null,
    val hasAudio: Boolean = false
)

data class FrameSample(
    val timestampMs: Long,
    val bitmap: Bitmap
)

// -------------------- Murf TTS Client (with state + error reporting) --------------------

class MurfTtsClient(
    private val context: Context,
    apiKey: String
) {
    @Volatile
    private var scope: CoroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    @Volatile
    private var sentenceChannel: Channel<String> = Channel(Channel.UNLIMITED)

    @Volatile
    private var activeTrack: AudioTrack? = null

    @Volatile
    private var cancelRequested: Boolean = false

    @Volatile
    private var config: TtsSettingsConfig = TtsSettingsConfig(
        apiKey = apiKey,
        voiceId = "Miles",
        model = "FALCON",
        locale = "en-UK",
        style = "Calm",
        sampleRate = 48000
    )

    // Status messages to surface errors / issues to the UI
    private val _statusMessages = Channel<String>(Channel.UNLIMITED)
    val statusMessages = _statusMessages.receiveAsFlow()

    // Whether TTS is currently playing
    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    init {
        startWorker()
    }

    fun updateConfig(newConfig: TtsSettingsConfig) {
        config = newConfig
    }

    private fun startWorker() {
        scope.launch {
            for (sentence in sentenceChannel) {
                if (sentence.isBlank()) continue
                try {
                    streamSentence(sentence)
                } catch (e: Exception) {
                    e.printStackTrace()
                    _statusMessages.trySend("Murf request error: ${e.message ?: "Unknown error"}")
                    _isSpeaking.value = false
                }
            }
        }
    }

    fun enqueueSentence(text: String) {
        val cleaned = text.trim()
        if (cleaned.isEmpty()) return

        cancelRequested = false
        sentenceChannel.trySend(cleaned)
    }

    suspend fun speak(text: String) {
        val (sentences, _) = splitCompletedSentences(text)
        sentences.forEach { s ->
            enqueueSentence(s)
        }
    }

    fun stopAll() {
        cancelRequested = true

        while (!sentenceChannel.isEmpty) {
            sentenceChannel.tryReceive().getOrNull() ?: break
        }

        try {
            activeTrack?.let { track ->
                try {
                    track.stop()
                } catch (_: Exception) {
                }
                try {
                    track.flush()
                } catch (_: Exception) {
                }
                try {
                    track.release()
                } catch (_: Exception) {
                }
            }
        } catch (_: Exception) {
        } finally {
            activeTrack = null
            _isSpeaking.value = false
        }
    }

    private suspend fun streamSentence(text: String) {
        val currentConfig = config

        if (currentConfig.apiKey.isBlank()) {
            _statusMessages.trySend(
                "Murf API key is empty. Open Settings → TTS Settings and enter your API key."
            )
            return
        }

        val jsonBody = JSONObject().apply {
            put("text", text)
            put("voiceId", currentConfig.voiceId)
            put("model", currentConfig.model)
            put("multiNativeLocale", currentConfig.locale)
            put("sampleRate", currentConfig.sampleRate.toString())
            put("style", currentConfig.style)
        }.toString()

        _isSpeaking.value = true
        try {
            withContext(Dispatchers.IO) {
                val url = URL("https://api.murf.ai/v1/speech/stream")
                val connection = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    doInput = true
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("api-key", currentConfig.apiKey)
                    connectTimeout = 15000
                    readTimeout = 30000
                }

                try {
                    connection.outputStream.use { os ->
                        val bytes = jsonBody.toByteArray(Charsets.UTF_8)
                        os.write(bytes)
                        os.flush()
                    }

                    val code = connection.responseCode
                    if (code !in 200..299) {
                        val errorText = try {
                            val stream = connection.errorStream
                            if (stream != null) {
                                stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                            } else {
                                null
                            }
                        } catch (_: Exception) {
                            null
                        }

                        val msg = buildString {
                            append("Murf request failed (HTTP $code). ")
                            append("Check your API key and TTS settings.")
                            if (!errorText.isNullOrBlank()) {
                                append(" Details: ")
                                append(errorText.take(300))
                            }
                        }

                        _statusMessages.trySend(msg)
                        connection.errorStream?.close()
                        return@withContext
                    }

                    val input = connection.inputStream

                    val header = ByteArray(44)
                    var headerRead = 0
                    while (headerRead < 44) {
                        val r = input.read(header, headerRead, 44 - headerRead)
                        if (r == -1) break
                        headerRead += r
                    }
                    if (headerRead < 44) {
                        input.close()
                        _statusMessages.trySend("Murf stream error: received incomplete WAV header.")
                        return@withContext
                    }

                    val bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                    bb.position(22)
                    val numChannels = bb.short.toInt()
                    val sampleRate = bb.int
                    bb.position(34)
                    val bitsPerSample = bb.short.toInt()

                    val channelConfig =
                        if (numChannels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
                    val audioEncoding =
                        if (bitsPerSample == 8) AudioFormat.ENCODING_PCM_8BIT else AudioFormat.ENCODING_PCM_16BIT

                    val minBufferSize = AudioTrack.getMinBufferSize(
                        sampleRate,
                        channelConfig,
                        audioEncoding
                    )
                    if (minBufferSize == AudioTrack.ERROR || minBufferSize == AudioTrack.ERROR_BAD_VALUE) {
                        input.close()
                        _statusMessages.trySend("Murf audio error: unsupported audio format or buffer size.")
                        return@withContext
                    }

                    val audioTrack = AudioTrack.Builder()
                        .setAudioAttributes(
                            AudioAttributes.Builder()
                                .setUsage(AudioAttributes.USAGE_MEDIA)
                                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                                .build()
                        )
                        .setAudioFormat(
                            AudioFormat.Builder()
                                .setEncoding(audioEncoding)
                                .setSampleRate(sampleRate)
                                .setChannelMask(channelConfig)
                                .build()
                        )
                        .setBufferSizeInBytes(minBufferSize * 2)
                        .setTransferMode(AudioTrack.MODE_STREAM)
                        .build()

                    activeTrack = audioTrack

                    try {
                        audioTrack.play()
                        val buffer = ByteArray(minBufferSize)
                        while (!cancelRequested) {
                            val read = input.read(buffer)
                            if (read <= 0) break
                            audioTrack.write(buffer, 0, read)
                        }
                        audioTrack.stop()
                    } finally {
                        try {
                            audioTrack.release()
                        } catch (_: Exception) {
                        }
                        activeTrack = null
                        input.close()
                    }
                } finally {
                    connection.disconnect()
                }
            }
        } finally {
            _isSpeaking.value = false
        }
    }
}

// -------------------- LLM Inference Manager (with state + clearContext) --------------------

class LlmInferenceManager(private val context: Context) {
    private var llmInference: LlmInference? = null
    private var llmSession: LlmInferenceSession? = null
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _partialResults = Channel<String>(Channel.UNLIMITED)
    val partialResults = _partialResults.receiveAsFlow()

    @Volatile
    private var generationToken: Int = 0

    @Volatile
    private var settings: LlmSettingsConfig = LlmSettingsConfig(
        systemPrompt = "",
        temperature = 0.8f,
        topP = 0.95f,
        topK = 40
    )

    // Whether the LLM is currently generating a response
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    fun updateSettings(newSettings: LlmSettingsConfig) {
        settings = newSettings
    }

    fun isInitialized(): Boolean = llmSession != null

    suspend fun loadModelFromUri(uri: Uri) {
        withContext(Dispatchers.IO) {
            val originalName = getFileName(context, uri) ?: "llm_model.task"
            val destinationFile = File(context.filesDir, originalName)
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destinationFile).use { output ->
                    input.copyTo(output)
                }
            }
            initialize(destinationFile.absolutePath)
        }
    }

    suspend fun initialize(modelPath: String) {
        withContext(Dispatchers.IO) {
            if (File(modelPath).exists()) {
                _partialResults.trySend("[System] Initializing model... (Path: $modelPath)")

                val gpuSuccess = tryInitialize(modelPath, LlmInference.Backend.GPU)
                if (!gpuSuccess) {
                    _partialResults.trySend("[System] GPU init failed. Trying CPU...")
                    val cpuSuccess = tryInitialize(modelPath, LlmInference.Backend.CPU)
                    if (!cpuSuccess) {
                        _partialResults.trySend("[System] Failed to initialize model.")
                    }
                }
            } else {
                _partialResults.trySend("[System] Model file not found at $modelPath")
            }
        }
    }

    private fun tryInitialize(modelPath: String, backend: LlmInference.Backend): Boolean {
        llmSession?.close()
        llmSession = null
        llmInference?.close()
        llmInference = null

        return try {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(4096)
                .setPreferredBackend(backend)
                .setMaxNumImages(10)
                .setAudioModelOptions(
                    AudioModelOptions.builder().build()
                )
                .build()

            val inference = LlmInference.createFromOptions(context, options)
            llmInference = inference

            recreateSession(inference)

            _partialResults.trySend("[System] Multimodal Session initialized successfully using $backend.")
            true
        } catch (e: Exception) {
            e.printStackTrace()
            _partialResults.trySend("[System] Error initializing with $backend: ${e.message}")
            false
        }
    }

    private fun recreateSession(inference: LlmInference) {
        try {
            llmSession?.close()
        } catch (_: Exception) {
        }
        llmSession = null

        val graphOptions = GraphOptions.builder()
            .setEnableVisionModality(true)
            .setEnableAudioModality(true)
            .build()

        val currentSettings = settings

        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions
            .builder()
            .setTemperature(currentSettings.temperature)
            .setTopK(currentSettings.topK)
            .setGraphOptions(graphOptions)
            .build()

        llmSession = LlmInferenceSession.createFromOptions(inference, sessionOptions)
    }

    // Clear model's conversation context but keep model + settings
    fun clearContext() {
        val inference = llmInference ?: run {
            _partialResults.trySend("[System] No model loaded; nothing to clear.")
            return
        }

        generationToken++
        _isGenerating.value = false

        try {
            recreateSession(inference)
            _partialResults.trySend("[System] Context cleared.")
        } catch (e: Exception) {
            e.printStackTrace()
            _partialResults.trySend("[System] Failed to clear context: ${e.message}")
        }
    }

    fun generateResponse(
        prompt: String,
        images: List<Bitmap> = emptyList(),
        audioBytes: ByteArray? = null
    ) {
        val session = llmSession
        if (session == null) {
            _partialResults.trySend("[System] Error: Session not initialized.")
            return
        }

        val token = ++generationToken
        val currentSettings = settings

        _isGenerating.value = true

        try {
            val fullPrompt = when {
                currentSettings.systemPrompt.isNotBlank() && prompt.isNotBlank() ->
                    currentSettings.systemPrompt + "\n\n" + prompt

                currentSettings.systemPrompt.isNotBlank() ->
                    currentSettings.systemPrompt

                else -> prompt
            }

            session.addQueryChunk(fullPrompt)

            if (images.isNotEmpty()) {
                for (bmp in images) {
                    val mpImage = BitmapImageBuilder(bmp).build()
                    session.addImage(mpImage)
                }
            }

            if (audioBytes != null) {
                session.addAudio(audioBytes)
            }

            session.generateResponseAsync { partialResult, isDone ->
                scope.launch {
                    if (token != generationToken) {
                        return@launch
                    }
                    _partialResults.send(partialResult)
                    if (isDone == true) {
                        _isGenerating.value = false
                    }
                }
            }
        } catch (e: Exception) {
            _isGenerating.value = false
            scope.launch {
                _partialResults.send("[Error] Generation failed: ${e.message}")
            }
        }
    }

    fun cancelGeneration() {
        generationToken++
        _isGenerating.value = false
        try {
            llmSession?.cancelGenerateResponseAsync()
        } catch (_: Exception) {
        }
    }

    fun close() {
        llmSession?.close()
        llmSession = null
        llmInference?.close()
        llmInference = null
        _isGenerating.value = false
        _partialResults.trySend("[System] Model unloaded.")
    }
}

// -------------------- Chat Screen --------------------

@Composable
fun SimpleChatUI(
    context: Context,
    inferenceManager: LlmInferenceManager,
    ttsClient: MurfTtsClient,
    messages: MutableList<Message>,
    systemMessages: MutableList<String>,
    onNavigateToVideo: () -> Unit,
    onOpenSettings: () -> Unit
) {
    var text by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    var isLoading by remember { mutableStateOf(false) }
    var selectedImage by remember { mutableStateOf<Bitmap?>(null) }

    var isRecording by remember { mutableStateOf(false) }
    var audioRecorder by remember { mutableStateOf<AudioRecorder?>(null) }

    var showUnloadDialog by remember { mutableStateOf(false) }
    var showClearContextDialog by remember { mutableStateOf(false) }

    // Reactive generation / TTS state
    val isGenerating by inferenceManager.isGenerating.collectAsState(initial = false)
    val isSpeaking by ttsClient.isSpeaking.collectAsState(initial = false)
    val isBusy = isGenerating || isSpeaking

    // List state for auto-scroll
    val listState = rememberLazyListState()

    // Are we at (or very near) the bottom?
    val isAtBottom by remember {
        derivedStateOf {
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index
            if (messages.isEmpty()) true
            else lastVisible == null || lastVisible >= messages.lastIndex
        }
    }

    // Auto-scroll when new messages are added (user/system)
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty() && isAtBottom) {
            listState.scrollToItem(messages.lastIndex)
        }
    }

    val modelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
        onResult = { uri: Uri? ->
            uri?.let {
                scope.launch(Dispatchers.IO) {
                    isLoading = true
                    withContext(Dispatchers.Main) {
                        systemMessages.add("Copying model file...")
                    }
                    try {
                        inferenceManager.loadModelFromUri(it)
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            systemMessages.add("Error: ${e.localizedMessage}")
                        }
                    } finally {
                        withContext(Dispatchers.Main) {
                            isLoading = false
                        }
                    }
                }
            }
        }
    )

    val imageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri: Uri? ->
            uri?.let {
                val inputStream = context.contentResolver.openInputStream(it)
                selectedImage = BitmapFactory.decodeStream(inputStream)
            }
        }
    )

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _: Boolean -> }

    // Collect LLM partial results
    LaunchedEffect(inferenceManager, ttsClient, listState) {
        var sentenceBuffer = ""
        inferenceManager.partialResults.collect { partial ->
            if (partial.startsWith("[System]") || partial.startsWith("[Error]")) {
                val cleaned = partial
                    .removePrefix("[System]")
                    .removePrefix("[Error]")
                    .trim()
                systemMessages.add(cleaned)
                if (isAtBottom && messages.isNotEmpty()) {
                    listState.scrollToItem(messages.lastIndex)
                }
                return@collect
            }

            if (messages.isNotEmpty() && !messages.last().isUser) {
                val lastMsg = messages.last()
                val index = messages.lastIndex
                if (lastMsg.isTyping) {
                    messages[index] = messages[index].copy(text = partial, isTyping = false)
                } else {
                    messages[index] = messages[index].copy(
                        text = messages[index].text + partial
                    )
                }
            } else {
                messages.add(Message(partial, false))
            }

            if (isAtBottom && messages.isNotEmpty()) {
                listState.scrollToItem(messages.lastIndex)
            }

            sentenceBuffer += partial
            val (sentences, remainder) = splitCompletedSentences(sentenceBuffer)
            sentenceBuffer = remainder
            sentences.forEach { s ->
                ttsClient.enqueueSentence(s)
            }
        }
    }

    // Collect Murf status/error messages
    LaunchedEffect(ttsClient, listState) {
        ttsClient.statusMessages.collect { msg ->
            systemMessages.add(msg)
            if (isAtBottom && messages.isNotEmpty()) {
                listState.scrollToItem(messages.lastIndex)
            }
        }
    }

    val isModelLoaded = inferenceManager.isInitialized()

    if (showUnloadDialog && isModelLoaded) {
        AlertDialog(
            onDismissRequest = { showUnloadDialog = false },
            title = { Text("Unload Model") },
            text = { Text("Do you want to unload the current model?") },
            confirmButton = {
                TextButton(onClick = {
                    inferenceManager.close()
                    ttsClient.stopAll()
                    showUnloadDialog = false
                }) {
                    Text("Unload")
                }
            },
            dismissButton = {
                TextButton(onClick = { showUnloadDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (showClearContextDialog && isModelLoaded) {
        AlertDialog(
            onDismissRequest = { showClearContextDialog = false },
            title = { Text("Reset Conversation Context") },
            text = { Text("Do you want to clear the current conversation context? This will reset history but keep all settings.") },
            confirmButton = {
                TextButton(onClick = {
                    inferenceManager.clearContext()
                    messages.clear()
                    systemMessages.add("Context cleared.")
                    showClearContextDialog = false
                }) {
                    Text("Reset")
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearContextDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Top bar: menu, model, video, clear context icon
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onOpenSettings) {
                    Icon(
                        imageVector = Icons.Default.Menu,
                        contentDescription = "Settings"
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = {
                        if (isModelLoaded) {
                            showUnloadDialog = true
                        } else {
                            modelLauncher.launch(arrayOf("*/*"))
                        }
                    },
                    enabled = !isLoading
                ) {
                    Text(
                        when {
                            isLoading -> "Loading..."
                            isModelLoaded -> "Model Loaded"
                            else -> "Load .litertlm"
                        }
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(
                    onClick = onNavigateToVideo,
                    enabled = isModelLoaded
                ) {
                    Text("Video Chat")
                }
                Spacer(modifier = Modifier.width(4.dp))
                IconButton(
                    onClick = {
                        if (isModelLoaded) {
                            showClearContextDialog = true
                        }
                    },
                    enabled = isModelLoaded
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Clear Context"
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // System feedback panel (scrollable, height-limited)
        if (systemMessages.isNotEmpty()) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth(),
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(8.dp)
            ) {
                val scrollState = rememberScrollState()
                Column(
                    modifier = Modifier
                        .padding(8.dp)
                        .heightIn(max = 120.dp)
                        .verticalScroll(scrollState),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "System",
                        style = MaterialTheme.typography.labelMedium
                    )
                    systemMessages.forEach { msg ->
                        Text(text = "• $msg", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { message ->
                MessageBubble(
                    message = message,
                    onSpeak = if (!message.isUser && message.text.isNotBlank()) {
                        {
                            scope.launch {
                                ttsClient.speak(message.text)
                            }
                        }
                    } else null
                )
            }
        }

        if (selectedImage != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Image(
                    bitmap = selectedImage!!.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier.size(60.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = { selectedImage = null }) { Text("X") }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { imageLauncher.launch("image/*") }) {
                Icon(Icons.Default.Add, contentDescription = "Add Image")
            }

            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = {
                    Text(if (isRecording) "Recording..." else "Message...")
                },
                enabled = isModelLoaded && !isRecording
            )

            IconButton(onClick = {
                if (!isModelLoaded) return@IconButton

                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    if (isRecording) {
                        isRecording = false
                        val audioData = audioRecorder?.stopRecording()
                        audioRecorder = null

                        if (audioData != null) {
                            messages.add(
                                Message(
                                    "[Audio Sent]",
                                    true,
                                    hasAudio = true
                                )
                            )
                            messages.add(Message("", false, isTyping = true))

                            inferenceManager.generateResponse(
                                prompt = text,
                                images = selectedImage?.let { listOf(it) } ?: emptyList(),
                                audioBytes = audioData
                            )

                            text = ""
                            selectedImage = null
                        }
                    } else {
                        isRecording = true
                        audioRecorder = AudioRecorder(context)
                        audioRecorder?.startRecording()
                    }
                } else {
                    permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            }) {
                if (isRecording) {
                    Text("Stop", color = MaterialTheme.colorScheme.error)
                } else {
                    Text("Mic")
                }
            }

            Button(
                onClick = {
                    if (!isModelLoaded) return@Button

                    if (isBusy) {
                        // Act as Stop button
                        inferenceManager.cancelGeneration()
                        ttsClient.stopAll()
                        return@Button
                    }

                    if (text.isNotBlank() || selectedImage != null) {
                        val userPrompt = text
                        messages.add(
                            Message(
                                userPrompt,
                                true,
                                image = selectedImage
                            )
                        )

                        val img = selectedImage
                        val prompt = userPrompt

                        text = ""
                        selectedImage = null

                        messages.add(Message("", false, isTyping = true))
                        inferenceManager.generateResponse(
                            prompt = prompt,
                            images = img?.let { listOf(it) } ?: emptyList(),
                            audioBytes = null
                        )
                    }
                },
                enabled = isModelLoaded && !isRecording
            ) {
                Text(if (isBusy) "Stop" else "Send")
            }
        }
    }
}

// -------------------- Video Chat Screen --------------------

@Composable
fun VideoChatUI(
    context: Context,
    inferenceManager: LlmInferenceManager,
    ttsClient: MurfTtsClient,
    messages: MutableList<Message>,
    systemMessages: MutableList<String>,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember { PreviewView(context) }

    var isRecording by remember { mutableStateOf(false) }
    var audioRecorder by remember { mutableStateOf<AudioRecorder?>(null) }
    var audioStartMs by remember { mutableStateOf(0L) }
    var audioEndMs by remember { mutableStateOf(0L) }

    val frameBuffer = remember { mutableStateListOf<FrameSample>() }
    var frameCaptureJob by remember { mutableStateOf<Job?>(null) }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _: Boolean -> }

    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _: Boolean -> }

    var showHistory by remember { mutableStateOf(false) }

    // List state for history auto-scroll
    val historyListState = rememberLazyListState()

    // Reactive generation / TTS state
    val isGenerating by inferenceManager.isGenerating.collectAsState(initial = false)
    val isSpeaking by ttsClient.isSpeaking.collectAsState(initial = false)
    val isBusy = isGenerating || isSpeaking

    // Auto-scroll history when visible and messages change
    LaunchedEffect(messages.size, showHistory) {
        if (showHistory && messages.isNotEmpty()) {
            historyListState.scrollToItem(messages.lastIndex)
        }
    }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().apply {
                    setSurfaceProvider(previewView.surfaceProvider)
                }
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview
                    )
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    // Collect LLM partial results (share messages/systemMessages, no System panel UI here)
    LaunchedEffect(inferenceManager, ttsClient) {
        var sentenceBuffer = ""
        inferenceManager.partialResults.collect { partial ->
            if (partial.startsWith("[System]") || partial.startsWith("[Error]")) {
                val cleaned = partial
                    .removePrefix("[System]")
                    .removePrefix("[Error]")
                    .trim()
                systemMessages.add(cleaned)
                return@collect
            }

            if (messages.isNotEmpty() && !messages.last().isUser) {
                val lastMsg = messages.last()
                val index = messages.lastIndex
                if (lastMsg.isTyping) {
                    messages[index] = messages[index].copy(text = partial, isTyping = false)
                } else {
                    messages[index] = messages[index].copy(
                        text = messages[index].text + partial
                    )
                }
            } else {
                messages.add(Message(partial, false))
            }

            sentenceBuffer += partial
            val (sentences, remainder) = splitCompletedSentences(sentenceBuffer)
            sentenceBuffer = remainder
            sentences.forEach { s ->
                ttsClient.enqueueSentence(s)
            }
        }
    }

    // Collect Murf status/error messages
    LaunchedEffect(ttsClient) {
        ttsClient.statusMessages.collect { msg ->
            systemMessages.add(msg)
        }
    }

    val isModelLoaded = inferenceManager.isInitialized()

    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .fillMaxWidth()
                .background(
                    MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)
                )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = onBack) {
                        Text("Back")
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = {
                            // Top Stop always stops generation and TTS
                            inferenceManager.cancelGeneration()
                            ttsClient.stopAll()
                        }
                    ) {
                        Text("Stop")
                    }
                    Spacer(modifier = Modifier.width(8.dp))

                    Surface(
                        shape = RoundedCornerShape(50),
                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                        tonalElevation = 4.dp,
                        modifier = Modifier
                            .widthIn(min = 40.dp)
                            .height(40.dp)
                            .clickable { showHistory = !showHistory }
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(if (showHistory) "Hide Chat" else "Show Chat")
                        }
                    }
                }
            }
        }

        if (showHistory) {
            Column(
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .padding(12.dp)
                    .widthIn(max = 220.dp)
                    .fillMaxHeight(0.6f)
                    .background(
                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                        shape = RoundedCornerShape(12.dp)
                    )
                    .padding(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "History",
                    style = MaterialTheme.typography.labelMedium
                )
                Divider()

                LazyColumn(
                    state = historyListState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(messages) { message ->
                        MessageBubble(
                            message = message,
                            onSpeak = if (!message.isUser && message.text.isNotBlank()) {
                                {
                                    scope.launch {
                                        ttsClient.speak(message.text)
                                    }
                                }
                            } else null
                        )
                    }
                }
            }
        }

        Button(
            onClick = {
                if (!isModelLoaded) {
                    systemMessages.add("Model not initialized. Load it first.")
                    return@Button
                }

                // If something is already running, let Hold Conversation act as Stop
                if (!isRecording && isBusy) {
                    inferenceManager.cancelGeneration()
                    ttsClient.stopAll()
                    return@Button
                }

                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.CAMERA
                    ) != PackageManager.PERMISSION_GRANTED
                ) {
                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    return@Button
                }
                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.RECORD_AUDIO
                    ) != PackageManager.PERMISSION_GRANTED
                ) {
                    micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    return@Button
                }

                if (!isRecording) {
                    isRecording = true
                    audioRecorder = AudioRecorder(context)
                    audioRecorder?.startRecording()
                    audioStartMs = SystemClock.elapsedRealtime()
                    frameBuffer.clear()

                    frameCaptureJob?.cancel()
                    frameCaptureJob = scope.launch {
                        while (isActive) {
                            val ts = SystemClock.elapsedRealtime()
                            withContext(Dispatchers.Main) {
                                val bmp = previewView.bitmap
                                if (bmp != null) {
                                    val scaled = Bitmap.createScaledBitmap(
                                        bmp,
                                        320,
                                        (320 * bmp.height / bmp.width.toFloat())
                                            .toInt()
                                            .coerceAtLeast(1),
                                        true
                                    )
                                    frameBuffer.add(FrameSample(ts, scaled))
                                }
                            }
                            delay(300L)
                        }
                    }
                } else {
                    isRecording = false
                    val audioData = audioRecorder?.stopRecording()
                    audioRecorder = null
                    audioEndMs = SystemClock.elapsedRealtime()

                    frameCaptureJob?.cancel()
                    frameCaptureJob = null

                    if (audioData != null) {
                        val framesCopy = frameBuffer.toList()
                        frameBuffer.clear()

                        val lastFrame = pickLastFrameForUtterance(
                            frames = framesCopy,
                            startMs = audioStartMs,
                            endMs = audioEndMs
                        )

                        // Store the image in the shared messages so the main chat UI can show it
                        messages.add(
                            Message(
                                text = "[Audio Sent]",
                                isUser = true,
                                image = lastFrame,
                                hasAudio = true
                            )
                        )
                        messages.add(Message("", isUser = false, isTyping = true))

                        scope.launch {
                            inferenceManager.generateResponse(
                                prompt = "",
                                images = lastFrame?.let { listOf(it) } ?: emptyList(),
                                audioBytes = audioData
                            )
                        }
                    }
                }
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
        ) {
            Text(
                when {
                    isRecording -> "Stop & Ask"
                    isBusy -> "Stop"
                    else -> "Hold Conversation"
                }
            )
        }
    }
}

// -------------------- Shared UI --------------------

@Composable
fun MessageBubble(
    message: Message,
    onSpeak: (() -> Unit)? = null
) {
    val bubbleColor =
        if (message.isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val textColor =
        if (message.isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    val alignment =
        if (message.isUser) Alignment.CenterEnd else Alignment.CenterStart

    val displayText = remember(message.text) { message.text.trimEnd() }

    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = alignment
    ) {
        Column(
            modifier = Modifier
                .background(color = bubbleColor, shape = RoundedCornerShape(8.dp))
                .padding(12.dp)
        ) {
            if (message.image != null) {
                Image(
                    bitmap = message.image.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .size(150.dp)
                        .padding(bottom = 8.dp)
                )
            }
            if (message.hasAudio) {
                Text("[Audio Clip]", color = textColor)
            }
            if (displayText.isNotEmpty()) {
                Text(text = displayText, color = textColor)
            }
            if (!message.isUser && onSpeak != null && displayText.isNotBlank()) {
                Spacer(modifier = Modifier.height(4.dp))
                TextButton(onClick = onSpeak) {
                    Text("Speak")
                }
            }
        }
    }
}

// -------------------- Audio Recorder --------------------

class AudioRecorder(private val context: Context) {
    private var recorder: AudioRecord? = null
    private var isRecording = false
    private val sampleRate = 16000
    private var outputStream = ByteArrayOutputStream()

    fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            return
        }

        if (ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        try {
            outputStream = ByteArrayOutputStream()

            recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
            recorder?.startRecording()
            isRecording = true
            Thread {
                val buffer = ByteArray(bufferSize)
                while (isRecording) {
                    val read = recorder?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        outputStream.write(buffer, 0, read)
                    }
                }
            }.start()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun stopRecording(): ByteArray? {
        isRecording = false
        try {
            recorder?.stop()
            recorder?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        recorder = null

        val pcmData = outputStream.toByteArray()
        return if (pcmData.isNotEmpty()) addWavHeader(pcmData) else null
    }

    private fun addWavHeader(pcmData: ByteArray): ByteArray {
        val numChannels = 1
        val bitsPerSample: Short = 16
        val byteRate = sampleRate * numChannels * bitsPerSample / 8
        val blockAlign: Short = (numChannels * bitsPerSample / 8).toShort()
        val totalDataLen = pcmData.size + 36

        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(totalDataLen)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)
        header.putShort(numChannels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(blockAlign)
        header.putShort(bitsPerSample)
        header.put("data".toByteArray())
        header.putInt(pcmData.size)

        return ByteArrayOutputStream().apply {
            write(header.array())
            write(pcmData)
        }.toByteArray()
    }
}

// -------------------- Helpers --------------------

fun getFileName(context: Context, uri: Uri): String? {
    var result: String? = null
    if (uri.scheme == "content") {
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        try {
            if (cursor != null && cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    result = cursor.getString(index)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            cursor?.close()
        }
    }
    if (result == null) {
        result = uri.path
        val cut = result?.lastIndexOf('/')
        if (cut != null && cut != -1) {
            result = result.substring(cut + 1)
        }
    }
    return result
}

// Sentence splitter: (completedSentences, remainder)
fun splitCompletedSentences(text: String): Pair<List<String>, String> {
    val sentences = mutableListOf<String>()
    var start = 0

    // Western + Japanese sentence terminators
    val terminators = setOf('.', '!', '?', '。', '！', '？', '\n')

    for (i in text.indices) {
        val c = text[i]
        if (c in terminators) {
            val end = i + 1
            val raw = text.substring(start, end)
            val sentence = raw.trim()
            if (sentence.isNotEmpty()) {
                sentences.add(sentence)
            }
            start = end
        }
    }

    val remainder = if (start < text.length) text.substring(start) else ""
    return sentences to remainder
}

// Pick only the last frame in the utterance window
fun pickLastFrameForUtterance(
    frames: List<FrameSample>,
    startMs: Long,
    endMs: Long
): Bitmap? {
    if (frames.isEmpty() || endMs <= startMs) return null
    val windowFrames = frames.filter { it.timestampMs in startMs..endMs }
    val last = windowFrames.maxByOrNull { it.timestampMs }
    return last?.bitmap
}
