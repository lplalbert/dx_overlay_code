#include "overlay_internal.h"

// 全局日志开关（默认关闭）
bool g_enableLogging = false;

// 调试日志文件
static std::ofstream g_logFile;
static std::filesystem::path g_logFilePath;
static bool g_startupLoggingActive = false;

static std::uint64_t CounterDelta32(UINT current, UINT previous) {
    if (current >= previous) {
        return static_cast<std::uint64_t>(current - previous);
    }
    return (std::uint64_t{1} << 32) - previous + current;
}

static void AppendLogMessage(const char* msg) {
    if (!g_logFile.is_open()) {
        const std::filesystem::path fallbackPath =
            g_logFilePath.empty() ? std::filesystem::path("dx_overlay.log") : g_logFilePath;
        g_logFile.open(fallbackPath, std::ios::out | std::ios::app);
    }
    if (g_logFile.is_open()) {
        g_logFile << msg;
        g_logFile.flush();
    }
    OutputDebugStringA(msg);
}

void InitializeLogging(const std::wstring& exeDir, bool enableVerboseLogging) {
    g_enableLogging = enableVerboseLogging;
    g_startupLoggingActive = true;
    g_logFilePath = std::filesystem::path(exeDir) / "dx_overlay.log";

    if (g_logFile.is_open()) {
        g_logFile.close();
    }
    g_logFile.open(g_logFilePath, std::ios::out | std::ios::trunc);

    SYSTEMTIME localTime = {};
    GetLocalTime(&localTime);

    char header[512];
    sprintf_s(header,
        "=== DX Overlay Startup Log ===\n"
        "Timestamp: %04u-%02u-%02u %02u:%02u:%02u.%03u\n"
        "Verbose runtime logging: %s\n",
        localTime.wYear, localTime.wMonth, localTime.wDay,
        localTime.wHour, localTime.wMinute, localTime.wSecond, localTime.wMilliseconds,
        g_enableLogging ? "enabled" : "disabled");
    AppendLogMessage(header);
}

void WriteStartupLog(const char* msg) {
    AppendLogMessage(msg);
}

void CompleteStartupLogging() {
    if (!g_startupLoggingActive) {
        return;
    }
    AppendLogMessage("=== Startup logging completed ===\n");
    g_startupLoggingActive = false;
}

void LogToFile(const char* msg) {
    if (!g_enableLogging && !g_startupLoggingActive) return;
    AppendLogMessage(msg);
}

// MSVC 特定的库链接指令（MinGW 链接参数由 build_release.bat 维护）
#ifdef _MSC_VER
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "dcomp.lib")
#endif

// HLSL 着色器代码
const char* g_vertexShaderCode = R"(
struct VS_INPUT {
    float2 Pos : POSITION;
    float2 Tex : TEXCOORD0;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

PS_INPUT VS(VS_INPUT input) {
    PS_INPUT output;
    output.Pos = float4(input.Pos, 0.0, 1.0);
    output.Tex = input.Tex;
    return output;
}
)";

const char* g_pixelShaderCode = R"(
// 使用 Texture2DArray 代替 Texture2D，性能优化
Texture2DArray texArray : register(t0);
SamplerState samPoint : register(s0);

// 常量缓冲区：独立的 Cb/Cr 强度 + 纹理相位
cbuffer AlphaBuffer : register(b0) {
    float staticAlpha;           // 静态 Cb 通道贴屏强度
    float dynamicAlpha;          // 动态 Cr 通道贴屏强度
    float textureIndex;          // 当前纹理索引
    float previousTextureIndex;  // 上一张纹理索引
    float transitionBlend;       // 过渡混合系数 [0, 1]
    float channelEncodedTemplates;
    float dynamicUsesSourceAlpha;
    float padding;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

float4 PS(PS_INPUT input) : SV_Target {
    float3 white = float3(1.0f, 1.0f, 1.0f);
    // 新版模板是通道数据纹理：R=Y、G=Cr(动态)、B=Cb(静态)、A=254。
    // 通道在数据空间中混合，避免 RGB 转换/裁剪让静态与动态信号串扰。
    float4 currentColor = texArray.Sample(samPoint, float3(input.Tex, textureIndex));
    float4 previousColor = texArray.Sample(samPoint, float3(input.Tex, previousTextureIndex));
    float blend = saturate(transitionBlend);
    float4 dynamicColor = lerp(previousColor, currentColor, blend);

    if (channelEncodedTemplates > 0.5f) {
        float outputAlpha = max(saturate(staticAlpha), saturate(dynamicAlpha));
        if (outputAlpha <= 0.0f) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        const float chromaNeutral = 128.0f / 255.0f;
        float staticWeight = saturate(staticAlpha) / outputAlpha;
        float dynamicWeight = saturate(dynamicAlpha) / outputAlpha;
        float y = dynamicColor.r;
        float cr = (dynamicColor.g - chromaNeutral) * dynamicWeight;
        float cb = (dynamicColor.b - chromaNeutral) * staticWeight;

        // Full-range YCrCb -> RGB. With equal strengths this recreates the
        // original combined template; unequal strengths independently scale
        // only the corresponding chroma contribution.
        float3 decodedColor;
        decodedColor.r = y + 1.402000f * cr;
        decodedColor.g = y - 0.714136f * cr - 0.344136f * cb;
        decodedColor.b = y + 1.772000f * cb;
        decodedColor = saturate(decodedColor);
        return float4(decodedColor * outputAlpha, outputAlpha);
    }

    // 兼容旧的 RGB(A) 模板；旧格式已经合并通道，因此只能使用较大的强度。
    float dynamicMask = dynamicUsesSourceAlpha > 0.5f
        ? saturate(dynamicColor.a)
        : saturate(max(abs(dynamicColor.r - white.r), max(abs(dynamicColor.g - white.g), abs(dynamicColor.b - white.b))) * 2.0f);
    float legacyAlpha = max(saturate(staticAlpha), saturate(dynamicAlpha)) * dynamicMask;
    float3 dynamicPremul = dynamicColor.rgb * legacyAlpha;

    return float4(dynamicPremul, legacyAlpha);
}
)";
DXOverlay::DXOverlay() : m_hInstance(GetModuleHandle(nullptr)) {
    m_lastError = "";
}

bool DXOverlay::InitializeTelemetry() {
    ShutdownTelemetry();

    m_telemetryMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(sizeof(DXOverlayTelemetry)),
        kDXOverlayTelemetryMappingName);
    if (!m_telemetryMapping) {
        return false;
    }

    m_telemetry = static_cast<DXOverlayTelemetry*>(MapViewOfFile(
        m_telemetryMapping,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(DXOverlayTelemetry)));
    if (!m_telemetry) {
        CloseHandle(m_telemetryMapping);
        m_telemetryMapping = nullptr;
        return false;
    }

    ZeroMemory(m_telemetry, sizeof(DXOverlayTelemetry));
    m_telemetry->magic = kDXOverlayTelemetryMagic;
    m_telemetry->version = kDXOverlayTelemetryVersion;
    m_telemetry->structSize = sizeof(DXOverlayTelemetry);
    m_telemetry->processId = GetCurrentProcessId();
    QueryPerformanceFrequency(&m_telemetry->qpcFrequency);
    PublishTelemetryState(false);
    return true;
}

void DXOverlay::ShutdownTelemetry() {
    if (m_telemetry) {
        InterlockedExchange(&m_telemetry->running, 0);
        UnmapViewOfFile(m_telemetry);
        m_telemetry = nullptr;
    }
    if (m_telemetryMapping) {
        CloseHandle(m_telemetryMapping);
        m_telemetryMapping = nullptr;
    }
}

void DXOverlay::PublishTelemetryState(bool running) {
    if (!m_telemetry) {
        return;
    }

    const LONG staticAlphaPpm = static_cast<LONG>(std::lround(
        std::clamp(m_staticAlpha, 0.0f, 1.0f) * 1000000.0f));
    const LONG dynamicAlphaPpm = static_cast<LONG>(std::lround(
        std::clamp(m_dynamicAlpha, 0.0f, 1.0f) * 1000000.0f));
    InterlockedExchange(&m_telemetry->shaderAlphaPpm,
        std::max(staticAlphaPpm, dynamicAlphaPpm));
    InterlockedExchange(&m_telemetry->staticAlphaPpm, staticAlphaPpm);
    InterlockedExchange(&m_telemetry->dynamicAlphaPpm, dynamicAlphaPpm);
    InterlockedExchange(&m_telemetry->currentTextureIndex, m_currentTextureIndex);
    InterlockedExchange(&m_telemetry->textureCount, m_textureCount);
    InterlockedExchange(&m_telemetry->screenWidth, m_screenWidth);
    InterlockedExchange(&m_telemetry->screenHeight, m_screenHeight);
    InterlockedExchange(&m_telemetry->refreshRate, m_refreshRate);
    InterlockedExchange(&m_telemetry->dynamicUsesSourceAlpha,
        m_dynamicTexturesUseSourceAlpha ? 1 : 0);
    InterlockedExchange(&m_telemetry->channelEncodedTemplates,
        m_channelEncodedTemplates ? 1 : 0);
    InterlockedExchange(&m_telemetry->holdFrames, m_targetTextureHoldFrames);
    InterlockedExchange(&m_telemetry->transitionFrames, m_transitionDurationFrames);
    InterlockedExchange64(
        &m_telemetry->phaseRefreshCorrection,
        static_cast<LONG64>(m_phaseRefreshCorrection));
    InterlockedExchange(&m_telemetry->running, running ? 1 : 0);
}

void DXOverlay::PublishPresentResult(
    HRESULT result,
    IDXGISwapChain1* swapChain,
    int submittedTextureIndex,
    std::uint64_t submittedPhaseOrdinal) {
    if (!m_telemetry) {
        return;
    }

    InterlockedExchange(&m_telemetry->lastPresentHresult, static_cast<LONG>(result));
    if (FAILED(result)) {
        InterlockedIncrement64(&m_telemetry->failedPresents);
        return;
    }

    LARGE_INTEGER presentTime = {};
    QueryPerformanceCounter(&presentTime);

    UINT presentId = 0;
    HRESULT presentIdResult = E_FAIL;
    if (swapChain) {
        // This is DXGI's identifier for the most recent successful Present,
        // not the overlay's local loop counter.
        presentIdResult = swapChain->GetLastPresentCount(&presentId);
    }

    const LONG64 nextSequence = m_telemetry->successfulPresents + 1;
    const std::size_t historyIndex = static_cast<std::size_t>(nextSequence) %
        kDXOverlayPresentHistoryCapacity;
    InterlockedExchange64(
        &m_telemetry->presentQpcHistory[historyIndex],
        presentTime.QuadPart);
    InterlockedExchange64(
        &m_telemetry->presentIdHistory[historyIndex],
        SUCCEEDED(presentIdResult) ? static_cast<LONG64>(presentId) : -1);
    InterlockedExchange64(
        &m_telemetry->presentPhaseOrdinalHistory[historyIndex],
        static_cast<LONG64>(submittedPhaseOrdinal));
    InterlockedExchange(
        &m_telemetry->presentTextureIndexHistory[historyIndex],
        submittedTextureIndex);
    InterlockedExchange64(&m_telemetry->lastPresentQpc, presentTime.QuadPart);
    InterlockedExchange(&m_telemetry->currentTextureIndex, submittedTextureIndex);
    MemoryBarrier();
    InterlockedExchange64(&m_telemetry->successfulPresents, nextSequence);
}

void DXOverlay::PublishFrameStatistics(IDXGISwapChain1* swapChain) {
    if (!m_telemetry || !swapChain) {
        return;
    }

    DXGI_FRAME_STATISTICS statistics = {};
    const HRESULT result = swapChain->GetFrameStatistics(&statistics);
    InterlockedExchange(
        &m_telemetry->lastFrameStatisticsHresult,
        static_cast<LONG>(result));

    if (result == DXGI_ERROR_FRAME_STATISTICS_DISJOINT) {
        m_displayPhaseAnchorValid = false;
        InterlockedIncrement64(&m_telemetry->frameStatisticsDisjoint);
        return;
    }
    if (FAILED(result)) {
        InterlockedIncrement64(&m_telemetry->frameStatisticsFailures);
        return;
    }

    const LONG64 presentCount = static_cast<LONG64>(statistics.PresentCount);
    if (presentCount == m_telemetry->lastDisplayedPresentCount) {
        return;
    }

    if (m_displayPhaseAnchorValid) {
        const std::uint64_t presentDelta = CounterDelta32(
            statistics.PresentCount,
            m_lastPhasePresentId);
        const std::uint64_t refreshDelta = CounterDelta32(
            statistics.PresentRefreshCount,
            m_lastPhaseRefreshCount);

        // PresentCount advances with accepted Presents, while
        // PresentRefreshCount advances with physical vblanks. Their signed
        // difference is the number of phase steps needed to make the next
        // submitted texture follow the display clock again.
        if (presentDelta > 0 && refreshDelta > 0 &&
            presentDelta <= kDXOverlayPresentHistoryCapacity &&
            refreshDelta <= kDXOverlayPresentHistoryCapacity * 4) {
            std::int64_t correction =
                static_cast<std::int64_t>(refreshDelta) -
                static_cast<std::int64_t>(presentDelta);

            const std::int64_t currentOrdinal =
                static_cast<std::int64_t>(m_totalFrameCount) +
                m_phaseRefreshCorrection;
            if (currentOrdinal + correction < 0) {
                correction = -currentOrdinal;
            }

            if (correction != 0) {
                m_phaseRefreshCorrection += correction;
                const LONG64 absoluteCorrection = static_cast<LONG64>(
                    correction < 0 ? -correction : correction);
                InterlockedExchange64(
                    &m_telemetry->phaseRefreshCorrection,
                    static_cast<LONG64>(m_phaseRefreshCorrection));
                InterlockedIncrement64(&m_telemetry->phaseResynchronizations);
                InterlockedExchangeAdd64(
                    &m_telemetry->phaseCorrectionSteps,
                    absoluteCorrection);

                if (g_enableLogging) {
                    char msg[256];
                    sprintf_s(msg,
                        "Display-phase resync: presentDelta=%llu, refreshDelta=%llu, correction=%lld, accumulated=%lld\n",
                        static_cast<unsigned long long>(presentDelta),
                        static_cast<unsigned long long>(refreshDelta),
                        static_cast<long long>(correction),
                        static_cast<long long>(m_phaseRefreshCorrection));
                    LogToFile(msg);
                }
            }
        } else {
            // Implausible deltas indicate a statistics discontinuity that was
            // not surfaced as DXGI_ERROR_FRAME_STATISTICS_DISJOINT. Re-anchor
            // without applying an unbounded phase jump.
            m_displayPhaseAnchorValid = false;
        }
    }

    m_lastPhasePresentId = statistics.PresentCount;
    m_lastPhaseRefreshCount = statistics.PresentRefreshCount;
    m_displayPhaseAnchorValid = true;

    // Match the frame reported by DXGI to the texture index that was submitted
    // with that Present. A miss remains -1 so diagnostics do not invent a
    // displayed texture when the bounded history has already wrapped.
    LONG displayedTextureIndex = -1;
    const LONG64 lastSubmittedSequence = m_telemetry->successfulPresents;
    const LONG64 oldestSubmittedSequence = std::max<LONG64>(
        1,
        lastSubmittedSequence - static_cast<LONG64>(kDXOverlayPresentHistoryCapacity) + 1);
    for (LONG64 sequence = lastSubmittedSequence;
         sequence >= oldestSubmittedSequence;
         --sequence) {
        const std::size_t historyIndex = static_cast<std::size_t>(sequence) %
            kDXOverlayPresentHistoryCapacity;
        if (m_telemetry->presentIdHistory[historyIndex] == presentCount) {
            displayedTextureIndex =
                m_telemetry->presentTextureIndexHistory[historyIndex];
            break;
        }
    }

    const LONG64 nextSample = m_telemetry->frameStatisticsSamples + 1;
    const std::size_t historyIndex = static_cast<std::size_t>(nextSample) %
        kDXOverlayPresentHistoryCapacity;
    DXOverlayDisplayFrameSample& sample =
        m_telemetry->displayFrameHistory[historyIndex];
    InterlockedExchange64(&sample.presentCount, presentCount);
    InterlockedExchange64(
        &sample.presentRefreshCount,
        static_cast<LONG64>(statistics.PresentRefreshCount));
    InterlockedExchange64(
        &sample.syncRefreshCount,
        static_cast<LONG64>(statistics.SyncRefreshCount));
    InterlockedExchange64(&sample.syncQpc, statistics.SyncQPCTime.QuadPart);
    InterlockedExchange(&sample.textureIndex, displayedTextureIndex);

    InterlockedExchange64(
        &m_telemetry->lastDisplayedPresentCount,
        presentCount);
    InterlockedExchange64(
        &m_telemetry->lastPresentRefreshCount,
        static_cast<LONG64>(statistics.PresentRefreshCount));
    InterlockedExchange64(
        &m_telemetry->lastSyncRefreshCount,
        static_cast<LONG64>(statistics.SyncRefreshCount));
    InterlockedExchange64(
        &m_telemetry->lastSyncQpc,
        statistics.SyncQPCTime.QuadPart);
    MemoryBarrier();
    InterlockedExchange64(&m_telemetry->frameStatisticsSamples, nextSample);
}

void DXOverlay::SetSwitchingBehavior(int holdFrames, int transitionFrames) {
    m_targetTextureHoldFrames = std::max(1, holdFrames);
    m_transitionDurationFrames = std::max(1, transitionFrames);
}

void DXOverlay::TriggerFatalShutdown(const char* logReason, const char* userMessage) {
    bool expected = false;
    if (!m_fatalShutdownTriggered.compare_exchange_strong(expected, true)) {
        return;
    }

    char msg[1024];
    sprintf_s(msg, "Fatal shutdown triggered: %s\n", logReason ? logReason : "unknown");
    LogToFile(msg);

    if (userMessage && *userMessage) {
        m_lastError = userMessage;
    } else if (logReason && *logReason) {
        m_lastError = logReason;
    } else {
        m_lastError = "Overlay encountered a fatal rendering error and will exit.";
    }

    m_running = false;
    m_isSuspended.store(false);
    PublishTelemetryState(false);

    HWND hwnd = m_hwnd;
    if (hwnd && IsWindow(hwnd)) {
        ShowWindow(hwnd, SW_HIDE);
        m_hwnd = nullptr;
        DestroyWindow(hwnd);
    } else {
        PostQuitMessage(0);
    }

    MessageBoxA(
        nullptr,
        userMessage && *userMessage
            ? userMessage
            : "The overlay rendering pipeline failed repeatedly and the application will exit.",
        "DX Overlay",
        MB_OK | MB_ICONERROR | MB_TOPMOST);
}

DXOverlay::~DXOverlay() {
    Stop();
    ShutdownTelemetry();

    // 清理帧延迟等待对象
    if (m_frameLatencyWaitableObject) {
        CloseHandle(m_frameLatencyWaitableObject);
        m_frameLatencyWaitableObject = nullptr;
    }
    if (m_mpoFrameLatencyWaitableObject) {
        CloseHandle(m_mpoFrameLatencyWaitableObject);
        m_mpoFrameLatencyWaitableObject = nullptr;
    }

    if (m_hwnd) {
        DestroyWindow(m_hwnd);
        m_hwnd = nullptr;
    }
}

bool DXOverlay::Initialize(
    const std::vector<std::wstring>& imagePaths,
    float staticAlpha,
    float dynamicAlpha) {
    m_lastError.clear();
    m_staticAlpha = std::clamp(staticAlpha, 0.0f, 1.0f);
    m_dynamicAlpha = std::clamp(dynamicAlpha, 0.0f, 1.0f);

    LogToFile("=== DXOverlay::Initialize started ===\n");
    char strengthMsg[192];
    sprintf_s(
        strengthMsg,
        "Requested template strengths: static Cb=%.4f, dynamic Cr=%.4f\n",
        m_staticAlpha,
        m_dynamicAlpha);
    LogToFile(strengthMsg);

    // 枚举所有显示器
    LogToFile("Step 0: Enumerating monitors...\n");
    EnumerateMonitors();

    char monitorMsg[512];
    sprintf_s(monitorMsg, "Found %zu monitors:\n", m_monitors.size());
    LogToFile(monitorMsg);
    for (size_t i = 0; i < m_monitors.size(); ++i) {
        sprintf_s(monitorMsg, "  Monitor %zu: %dx%d at (%d,%d) %s\n",
                  i, m_monitors[i].width, m_monitors[i].height,
                  m_monitors[i].rect.left, m_monitors[i].rect.top,
                  m_monitors[i].isPrimary ? "[PRIMARY]" : "");
        LogToFile(monitorMsg);
    }

    // 选择目标显示器（默认主显示器）
    if (!SelectMonitor(m_targetMonitorIndex)) {
        // 如果指定的显示器无效，使用主显示器
        LogToFile("Target monitor not found, using primary monitor\n");
        SelectMonitor(0);
    }

    char stepMsg[256];
    sprintf_s(stepMsg, "Step 1: Selected monitor %d, Screen size: %dx%d at (%d, %d), refresh=%dHz\n",
              m_targetMonitorIndex, m_screenWidth, m_screenHeight, m_windowX, m_windowY, m_refreshRate);
    LogToFile(stepMsg);

    // 创建窗口
    LogToFile("Step 2: Creating overlay window...\n");
    if (!CreateOverlayWindow()) {
        LogToFile("ERROR: Failed to create overlay window\n");
        m_lastError = "Failed to create overlay window";
        return false;
    }
    LogToFile("Step 2: Overlay window created successfully\n");

    // 初始化 DirectX
    LogToFile("Step 3: Initializing DirectX...\n");
    if (!InitializeDirectX()) {
        LogToFile("ERROR: Failed to initialize DirectX\n");
        m_lastError = "Failed to initialize DirectX";
        return false;
    }
    LogToFile("Step 3: DirectX initialized successfully\n");

    // Step 3a: 检测 GPU 信息
    LogToFile("Step 3a: Detecting GPU information...\n");
    DetectGPUInfo();

    // 生产稳定路径固定使用标准 DirectComposition swap chain。
    // MPO 会扩大显卡驱动和显示器热插拔的恢复矩阵，暂不在运行时启用。
    m_mpoCapability = MPOCapability::NotSupported;
    m_useMPO = false;
    m_mpoInitialized = false;
    LogToFile("Step 3b: MPO disabled by stability policy; using standard DirectComposition path\n");

    // Step 3c: 应用 GPU 特定优化，并让演示参数真正落地
    LogToFile("Step 3c: Applying GPU-specific optimizations...\n");
    ApplyGPUOptimizations();

    if (m_standardSwapChainBufferCount != kSwapChainBufferCount || m_standardMaximumFrameLatency != 1u) {
        LogToFile("Step 3d: Reconfiguring standard swap chain with GPU-specific presentation settings...\n");
        if (!ResizeSwapChainBuffers(m_screenWidth, m_screenHeight, "apply gpu-specific presentation config")) {
            LogToFile("ERROR: Failed to apply GPU-specific presentation settings\n");
            m_lastError = "Failed to apply GPU-specific presentation settings";
            return false;
        }
    }

    // 创建着色器
    LogToFile("Step 4: Creating shaders...\n");
    if (!CreateShaders()) {
        LogToFile("ERROR: Failed to create shaders\n");
        m_lastError = "Failed to create shaders";
        return false;
    }
    LogToFile("Step 4: Shaders created successfully\n");

    // 加载纹理
    LogToFile("Step 5: Loading textures...\n");
    if (!LoadTextures(imagePaths)) {
        LogToFile("ERROR: Failed to load textures\n");
        if (m_lastError.empty()) {
            m_lastError = "Failed to load textures";
        }
        return false;
    }
    LogToFile("Step 5: Textures loaded successfully\n");

    // 初始化常量缓冲区（只需一次）
    UpdateConstantBuffer();
    LogToFile("Step 6: Constant buffer initialized\n");

    // 成功 Present 提供基础序号；显示统计用物理刷新差值自动修正相位。
    m_targetTextureHoldFrames = std::max(1, m_targetTextureHoldFrames);
    m_transitionDurationFrames = std::max(1, m_transitionDurationFrames);
    m_currentTextureIndex = 0;
    m_previousTextureIndex = 0;
    m_totalFrameCount = 0;
    m_phaseRefreshCorrection = 0;
    m_displayPhaseAnchorValid = false;
    m_lastPhasePresentId = 0;
    m_lastPhaseRefreshCount = 0;

    char switchMsg[256];
    sprintf_s(switchMsg,
        "Display-clock switching: refresh=%dHz, holdFrames=%d, transitionFrames=%d, nominalInterval=%dms\n",
        std::max(1, m_refreshRate),
        m_targetTextureHoldFrames,
        m_transitionDurationFrames,
        std::max(1, 1000 / std::max(1, m_refreshRate)));
    LogToFile(switchMsg);

    // MPO 仅保留代码作为未来的显式实验路径，生产默认不进入。
    if (kEnableMPO && m_mpoCapability != MPOCapability::NotSupported) {
        LogToFile("Step 7: Initializing MPO optimization...\n");
        if (InitializeMPOOptimization()) {
            LogToFile("Step 7: MPO optimization initialized successfully\n");
            m_useMPO = true;
        } else {
            LogToFile("Step 7: MPO initialization failed, falling back to standard rendering\n");
            m_useMPO = false;
        }
    }

    // 输出最终配置
    char finalMsg[768];
    sprintf_s(finalMsg, "=== Final Configuration ===\n"
                        "  GPU: %ls\n"
                        "  MPO: %s\n"
                        "  Render Mode: %s\n"
                        "  Policy Buffer Count: %d\n"
                        "  Standard Buffers: %u\n"
                        "  Standard Max Latency: %u\n"
                        "  MPO Buffers: %u\n"
                        "  MPO Max Latency: %u\n"
                        "  Low Latency: %s\n"
                        "  Static Cb Strength: %.4f\n"
                        "  Dynamic Cr Strength: %.4f\n"
                        "  Template Format: %s\n",
              m_gpuDescription.c_str(),
              m_useMPO ? "Enabled" : "Disabled",
              m_useMPO ? "MPO Optimized" : "Standard",
              m_optimalBufferCount,
              m_standardSwapChainBufferCount,
              m_standardMaximumFrameLatency,
              m_mpoSwapChainBufferCount,
              m_mpoMaximumFrameLatency,
              m_reduceLatency ? "Yes" : "No",
              m_staticAlpha,
              m_dynamicAlpha,
              m_channelEncodedTemplates ? "Independent Y/Cr/Cb data" : "Legacy combined RGB(A)");
    LogToFile(finalMsg);

    if (!InitializeTelemetry()) {
        LogToFile("WARNING: Runtime telemetry is unavailable; diagnostics will not attach\n");
    }
    PublishTelemetryState(false);
    LogToFile("All initialization steps completed successfully!\n");
    return true;
}

// 多显示器枚举回调函数
// 枚举所有显示器
// 获取显示器数量
// 选择指定显示器
