#include "overlay_internal.h"

DXOverlay::TexturePhase DXOverlay::CalculateTexturePhase(
    std::uint64_t successfulPresentOrdinal) const {
    TexturePhase phase;
    if (m_textureCount <= 1) {
        return phase;
    }

    const std::uint64_t holdFrames = static_cast<std::uint64_t>(
        std::max(1, m_targetTextureHoldFrames));
    const std::uint64_t transitionIntervals = static_cast<std::uint64_t>(
        std::max(0, m_transitionDurationFrames - 1));
    const std::uint64_t segmentLength = holdFrames + transitionIntervals;
    const std::uint64_t segment = successfulPresentOrdinal / segmentLength;
    const std::uint64_t segmentPosition = successfulPresentOrdinal % segmentLength;

    const int sourceIndex = static_cast<int>(
        segment % static_cast<std::uint64_t>(m_textureCount));
    const int targetIndex = (sourceIndex + 1) % m_textureCount;

    if (segmentPosition < holdFrames || transitionIntervals == 0) {
        phase.currentTextureIndex = sourceIndex;
        phase.previousTextureIndex = sourceIndex;
        phase.transitionBlend = 1.0f;
        return phase;
    }

    const std::uint64_t transitionStep = segmentPosition - holdFrames + 1;
    phase.currentTextureIndex = targetIndex;
    phase.previousTextureIndex = sourceIndex;
    phase.transitionBlend = static_cast<float>(transitionStep) /
        static_cast<float>(m_transitionDurationFrames);
    return phase;
}

void DXOverlay::UpdateConstantBuffer() {
    // 常量缓冲区只在初始化时设置一次，不再每帧更新
    // 纹理切换索引在 Render() 中直接计算
    if (m_textureCount == 0) return;  // 使用 m_textureCount 替代 m_textures.size()

    // 本地结构体与类成员结构体和 shader cbuffer 必须匹配
    ConstantBufferData cbLocal;
    cbLocal.staticAlpha = m_staticAlpha;
    cbLocal.dynamicAlpha = m_dynamicAlpha;
    cbLocal.textureIndex = 0.0f;
    cbLocal.previousTextureIndex = 0.0f;
    cbLocal.transitionBlend = 1.0f;
    cbLocal.channelEncodedTemplates = m_channelEncodedTemplates ? 1.0f : 0.0f;
    cbLocal.dynamicUsesSourceAlpha = m_dynamicTexturesUseSourceAlpha ? 1.0f : 0.0f;
    cbLocal.padding = 0.0f;

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(m_context->Map(m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, &cbLocal, sizeof(cbLocal));
        m_context->Unmap(m_constantBuffer.Get(), 0);
    }
}

void DXOverlay::Render() {
    if (!m_running || m_isSuspended.load()) return;

    if ((!m_device || !m_context) && !m_isRecovering.load()) {
        QueueRecovery(
            RecoveryRequest::Device,
            E_POINTER,
            "render loop lost Direct3D device/context");
        return;
    }

    // 根据 MPO 支持情况选择渲染路径
    if (m_useMPO && m_mpoInitialized) {
        RenderWithMPO();
    } else {
        RenderWithoutMPO();
    }
}

bool DXOverlay::RenderToSwapChain(
    IDXGISwapChain1* swapChain,
    ComPtr<ID3D11RenderTargetView>& renderTargetView,
    bool useRestrictedPresent,
    IDXGIOutput* restrictOutput,
    const char* label) {
    if (!m_running || !m_device || !m_context || !swapChain) {
        return false;
    }

    if (m_textureCount == 0) {
        return false;
    }

    // The waitable object is only a render-pacing permit; it is not proof that
    // a particular frame reached the monitor. Query DXGI's presentation
    // statistics so diagnostics can observe the last displayed Present ID.
    PublishFrameStatistics(swapChain);

    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(m_screenWidth);
    vp.Height = static_cast<float>(m_screenHeight);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    m_context->RSSetViewports(1, &vp);

    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    m_context->IASetVertexBuffers(0, 1, m_vertexBuffer.GetAddressOf(), &stride, &offset);
    m_context->IASetInputLayout(m_inputLayout.Get());
    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
    m_context->PSSetShader(m_pixelShader.Get(), nullptr, 0);
    m_context->PSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());

    ID3D11ShaderResourceView* srv = m_textureArraySRV.Get();
    ID3D11SamplerState* sampler = m_pointSamplerState.Get();
    m_context->PSSetShaderResources(0, 1, &srv);
    m_context->PSSetSamplers(0, 1, &sampler);
    m_context->OMSetDepthStencilState(nullptr, 0);
    m_context->RSSetState(nullptr);

    float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    m_context->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);

    ID3D11RenderTargetView* currentRTV = renderTargetView.Get();
    if (!currentRTV) {
        return false;
    }

    m_context->OMSetRenderTargets(1, &currentRTV, nullptr);

    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    m_context->ClearRenderTargetView(currentRTV, clearColor);

    const std::int64_t correctedPhaseOrdinal =
        static_cast<std::int64_t>(m_totalFrameCount) +
        m_phaseRefreshCorrection;
    const std::uint64_t phaseOrdinal = static_cast<std::uint64_t>(
        std::max<std::int64_t>(0, correctedPhaseOrdinal));
    const TexturePhase phase = CalculateTexturePhase(phaseOrdinal);

    if (m_textureArraySRV && m_textureCount > 0) {
        const float textureIndex = static_cast<float>(phase.currentTextureIndex);
        const float previousTextureIndex = static_cast<float>(phase.previousTextureIndex);
        const float transitionBlend = phase.transitionBlend;

        D3D11_MAPPED_SUBRESOURCE mapped;
        HRESULT hrMap = m_context->Map(m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (SUCCEEDED(hrMap)) {
            m_consecutiveMapFailures = 0;
            ConstantBufferData* cbData = reinterpret_cast<ConstantBufferData*>(mapped.pData);
            cbData->staticAlpha = m_staticAlpha;
            cbData->dynamicAlpha = m_dynamicAlpha;
            cbData->textureIndex = textureIndex;
            cbData->previousTextureIndex = previousTextureIndex;
            cbData->transitionBlend = transitionBlend;
            cbData->channelEncodedTemplates = m_channelEncodedTemplates ? 1.0f : 0.0f;
            cbData->dynamicUsesSourceAlpha = m_dynamicTexturesUseSourceAlpha ? 1.0f : 0.0f;
            cbData->padding = 0.0f;
            m_context->Unmap(m_constantBuffer.Get(), 0);
            if (m_telemetry) {
                const LONG staticAlphaPpm = static_cast<LONG>(std::lround(
                    std::clamp(m_staticAlpha, 0.0f, 1.0f) * 1000000.0f));
                const LONG dynamicAlphaPpm = static_cast<LONG>(std::lround(
                    std::clamp(m_dynamicAlpha, 0.0f, 1.0f) * 1000000.0f));
                InterlockedExchange(
                    &m_telemetry->shaderAlphaPpm,
                    std::max(staticAlphaPpm, dynamicAlphaPpm));
                InterlockedExchange(&m_telemetry->staticAlphaPpm, staticAlphaPpm);
                InterlockedExchange(&m_telemetry->dynamicAlphaPpm, dynamicAlphaPpm);
                InterlockedIncrement64(&m_telemetry->shaderConstantUpdates);
            }
            m_context->PSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());
        } else {
            m_consecutiveMapFailures++;
            if (m_consecutiveMapFailures >= 3) {
                QueueRecovery(
                    RecoveryRequest::Device,
                    hrMap,
                    "constant-buffer Map failed repeatedly");
                m_consecutiveMapFailures = 0;
            }
            return false;
        }
    } else if (g_enableLogging && m_totalFrameCount % 60 == 0) {
        char msg[128];
        sprintf_s(msg, "WARNING: Texture array SRV is null in %s render path\n",
            label ? label : "unknown");
        LogToFile(msg);
    }

    m_context->Draw(4, 0);

    HRESULT hrPresent = S_OK;
    if (useRestrictedPresent) {
        DXGI_PRESENT_PARAMETERS presentParams = {};
        UINT presentFlags = 0;
        if (restrictOutput) {
            presentFlags |= DXGI_PRESENT_RESTRICT_TO_OUTPUT;
        }
        hrPresent = swapChain->Present1(1, presentFlags, &presentParams);
    } else {
        hrPresent = swapChain->Present(1, 0);
    }
    PublishPresentResult(
        hrPresent,
        swapChain,
        phase.currentTextureIndex,
        phaseOrdinal);

    if (FAILED(hrPresent)) {
        m_consecutivePresentFailures++;

        char presentFailMsg[256];
        sprintf_s(
            presentFailMsg,
            "%s present failed consecutively %d/%d times (HRESULT: 0x%08X)\n",
            label ? label : "unknown",
            m_consecutivePresentFailures,
            m_maxConsecutivePresentFailures,
            hrPresent);
        LogToFile(presentFailMsg);

        if (hrPresent == DXGI_ERROR_DEVICE_REMOVED ||
            hrPresent == DXGI_ERROR_DEVICE_RESET ||
            hrPresent == DXGI_ERROR_DEVICE_HUNG ||
            hrPresent == DXGI_ERROR_DRIVER_INTERNAL_ERROR) {
            QueueRecovery(RecoveryRequest::Device, hrPresent, "Present reported device loss");
        } else if (hrPresent == DXGI_ERROR_INVALID_CALL ||
                   hrPresent == DXGI_ERROR_ACCESS_LOST ||
                   m_consecutivePresentFailures >= 3) {
            QueueRecovery(RecoveryRequest::Presentation, hrPresent, "Present failed repeatedly");
        }

        if (m_consecutivePresentFailures >= m_maxConsecutivePresentFailures) {
            char userMessage[768];
            sprintf_s(
                userMessage,
                "The overlay failed to present frames for %d consecutive attempts and will now exit.\n\nLast render path: %s\nHRESULT: 0x%08X",
                m_consecutivePresentFailures,
                label ? label : "unknown",
                hrPresent);
            TriggerFatalShutdown("Exceeded consecutive Present failure threshold", userMessage);
        }
    }

    static bool firstPresentLogged = false;
    if (!firstPresentLogged) {
        char msg[256];
        sprintf_s(msg, "First %s Present returned: 0x%08X\n",
            label ? label : "unknown", hrPresent);
        LogToFile(msg);
        firstPresentLogged = true;
    }

    if (SUCCEEDED(hrPresent)) {
        m_consecutivePresentFailures = 0;
        m_currentTextureIndex = phase.currentTextureIndex;
        m_previousTextureIndex = phase.previousTextureIndex;
        m_totalFrameCount++;
    }

    return SUCCEEDED(hrPresent);
}

void DXOverlay::RenderWithMPO() {
    if (!m_running || !m_device || !m_context) {
        return;
    }

    if (!m_swapChainMPO) {
        if (m_swapChain) {
            LogToFile("MPO swap chain missing, falling back to standard rendering path\n");
            m_useMPO = false;
            m_mpoInitialized = false;
            if (!SetActiveVisualContent(m_swapChain.Get(), "missing MPO swap chain fallback")) {
                return;
            }
            RenderWithoutMPO();
            return;
        }

        if (!m_isRecovering.load()) {
            TriggerFatalShutdown(
                "MPO rendering path lost all swap chains",
                "The overlay lost both MPO and standard rendering surfaces and will now exit.");
        }
        return;
    }

    static bool firstRender = true;
    if (firstRender) {
        LogToFile("=== MPO Rendering Path Enabled ===\n");
        if (m_isAMDAPU) {
            LogToFile("AMD APU optimized: Dedicated MPO swap chain with restricted output present\n");
        } else if (m_gpuVendor == GPUVendor::Intel && m_gpuType == GPUType::Integrated) {
            LogToFile("Intel iGPU optimized: Dedicated MPO swap chain with restricted output present\n");
        } else if (m_gpuVendor == GPUVendor::AMD) {
            LogToFile("AMD GPU optimized: Dedicated MPO swap chain with restricted output present\n");
        } else {
            LogToFile("Standard MPO: Dedicated swap chain, independent plane promotion path\n");
        }
        firstRender = false;
    }

    if (!RenderToSwapChain(
            m_swapChainMPO.Get(),
            m_mpoRenderTargetView,
            true,
            m_mpoOutput.Get(),
            "MPO")) {
        if (m_swapChain && !m_isRecovering.load()) {
            LogToFile("MPO render path failed, falling back to standard swap chain\n");
            m_useMPO = false;
            m_mpoInitialized = false;
            if (m_mpoFrameLatencyWaitableObject) {
                CloseHandle(m_mpoFrameLatencyWaitableObject);
                m_mpoFrameLatencyWaitableObject = nullptr;
            }
            m_swapChainMPO2.Reset();
            m_swapChainMPO.Reset();
            m_mpoRenderTargetView.Reset();
            if (!SetActiveVisualContent(m_swapChain.Get(), "MPO render fallback")) {
                return;
            }
        }
    }
}

void DXOverlay::RenderWithoutMPO() {
    if (!m_running || !m_device || !m_context) {
        return;
    }

    if (!m_swapChain) {
        if (!m_isRecovering.load()) {
            QueueRecovery(
                RecoveryRequest::Presentation,
                E_POINTER,
                "standard rendering path lost its swap chain");
        }
        return;
    }

    static bool firstRender = true;
    if (firstRender) {
        LogToFile("Standard rendering path initialized\n");
        firstRender = false;
    }

    RenderToSwapChain(
        m_swapChain.Get(),
        m_renderTargetView,
        false,
        nullptr,
        "standard");
}

HANDLE DXOverlay::GetActiveFrameLatencyWaitableObject() const {
    if (m_useMPO && m_mpoInitialized && m_mpoFrameLatencyWaitableObject) {
        return m_mpoFrameLatencyWaitableObject;
    }
    return m_frameLatencyWaitableObject;
}

void DXOverlay::Run() {
    m_running = true;
    PublishTelemetryState(true);
    LogToFile("Run() started, entering high-precision render loop...\n");

    // 让 MMCSS 接管渲染线程调度；不可用时再回退到传统优先级。
    int previousPriority = GetThreadPriority(GetCurrentThread());
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);

    MmcssRegistration mmcss;
    DWORD mmcssError = ERROR_SUCCESS;
    const char* mmcssProfileName = nullptr;
    bool mmcssActive = mmcss.Activate(L"Games", AVRT_PRIORITY_HIGH, &mmcssError);
    if (mmcssActive) {
        mmcssProfileName = "Games";
    }
    if (!mmcssActive) {
        mmcssActive = mmcss.Activate(L"Playback", AVRT_PRIORITY_HIGH, &mmcssError);
        if (mmcssActive) {
            mmcssProfileName = "Playback";
        }
    }

    if (mmcssActive) {
        char msg[160];
        sprintf_s(msg, "MMCSS enabled for render loop (%s)\n", mmcssProfileName ? mmcssProfileName : "unknown");
        LogToFile(msg);
    } else {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        char msg[160];
        sprintf_s(msg, "MMCSS unavailable, falling back to ABOVE_NORMAL priority (error=%lu)\n", mmcssError);
        LogToFile(msg);
    }

    MSG msg = {};
    int loopCount = 0;

    // 使用 VSync 控制帧率，Run 循环只需处理消息和调用 Render
    // Present(1, 0) 会自动同步到显示器刷新率
    LARGE_INTEGER freq, lastSecondTime, currentTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&lastSecondTime);
    int frameInSecond = 0;

    // 定期重新置顶窗口的计数器
    int topMostCheckCounter = 0;
    const int topMostCheckInterval = std::max(120, std::max(1, m_refreshRate) * 4);  // 约每4秒按需检查一次
    int consecutiveFrameLatencyTimeouts = 0;
    int consecutiveFrameLatencyFailures = 0;
    auto frameLatencyWaitBypassUntil = std::chrono::steady_clock::time_point::min();

    while (m_running) {
        bool shouldRender = true;

        // 联合等待 VSync 和窗口消息，避免显示变化、退出和恢复消息被帧等待阻塞。
        HANDLE activeWaitableObject = GetActiveFrameLatencyWaitableObject();
        if (std::chrono::steady_clock::now() < frameLatencyWaitBypassUntil) {
            activeWaitableObject = nullptr;
        }
        if (activeWaitableObject) {
            HANDLE waitHandles[] = { activeWaitableObject };
            DWORD waitResult = MsgWaitForMultipleObjectsEx(
                ARRAYSIZE(waitHandles),
                waitHandles,
                kFrameLatencyWaitTimeoutMs,
                QS_ALLINPUT,
                MWMO_INPUTAVAILABLE);
            if (waitResult == WAIT_TIMEOUT) {
                shouldRender = false;
                consecutiveFrameLatencyTimeouts++;
                consecutiveFrameLatencyFailures = 0;
                if (g_enableLogging) {
                    if (consecutiveFrameLatencyTimeouts == 1 ||
                        consecutiveFrameLatencyTimeouts % 30 == 0) {
                        char msg[128];
                        sprintf_s(msg, "WARNING: Frame latency wait timeout (count=%d)\n",
                                  consecutiveFrameLatencyTimeouts);
                        LogToFile(msg);
                    }
                }
                if (consecutiveFrameLatencyTimeouts >= kMaxFrameLatencyTimeoutsBeforeRecovery) {
                    QueueRecovery(
                        RecoveryRequest::Presentation,
                        HRESULT_FROM_WIN32(WAIT_TIMEOUT),
                        "frame-latency wait timed out repeatedly");
                    // 恢复期间改用 Present(1, 0) 自身的 VSync 阻塞，避免坏句柄导致永久冻结或频繁重建。
                    frameLatencyWaitBypassUntil =
                        std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    consecutiveFrameLatencyTimeouts = 0;
                }
            } else if (waitResult == WAIT_OBJECT_0) {
                consecutiveFrameLatencyTimeouts = 0;
                consecutiveFrameLatencyFailures = 0;
            } else if (waitResult == WAIT_OBJECT_0 + ARRAYSIZE(waitHandles)) {
                // 消息就绪，先处理消息，不强行提交一帧。
                shouldRender = false;
            } else {
                shouldRender = false;
                consecutiveFrameLatencyFailures++;
                consecutiveFrameLatencyTimeouts = 0;

                DWORD waitError = (waitResult == WAIT_FAILED) ? ::GetLastError() : ERROR_INVALID_HANDLE;
                char waitMsg[192];
                sprintf_s(waitMsg,
                    "ERROR: Frame latency wait failed (result=0x%08lX, error=%lu, consecutive=%d)\n",
                    waitResult,
                    waitError,
                    consecutiveFrameLatencyFailures);
                LogToFile(waitMsg);

                if (consecutiveFrameLatencyFailures >= kMaxFrameLatencyWaitFailuresBeforeRecovery) {
                    QueueRecovery(
                        RecoveryRequest::Presentation,
                        HRESULT_FROM_WIN32(waitError),
                        "frame-latency wait handle failed repeatedly");
                    frameLatencyWaitBypassUntil =
                        std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    consecutiveFrameLatencyFailures = 0;
                }
            }
        } else {
            consecutiveFrameLatencyTimeouts = 0;
            consecutiveFrameLatencyFailures = 0;
            // Present(1, 0) is the fallback pacer. Sleeping for a nominal
            // refresh interval here as well can miss the next vblank and cut
            // the effective rate roughly in half.
            shouldRender = true;
        }

        // 处理消息（非阻塞）
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                m_running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (!m_running) break;

        if (!ProcessQueuedRecovery()) {
            if (m_running) {
                Sleep(50);
            }
            continue;
        }

        // 按需确保窗口在顶层，避免频繁 SetWindowPos 干扰输入期的合成节奏
        topMostCheckCounter++;
        if (topMostCheckCounter >= topMostCheckInterval) {
            topMostCheckCounter = 0;
            CheckAndRestoreTopmost();
        }

        if (!shouldRender) {
            continue;
        }

        // 渲染帧 - Present(1, 0) 会等待 VSync
        Render();
        loopCount++;
        frameInSecond++;

        if (!ProcessQueuedRecovery()) {
            if (m_running) {
                Sleep(50);
            }
            continue;
        }

        // 每秒记录一次状态
        QueryPerformanceCounter(&currentTime);
        if (currentTime.QuadPart - lastSecondTime.QuadPart >= freq.QuadPart) {
            if (g_enableLogging) {
                char logMsg[256];
                sprintf_s(logMsg, "Frame %d, textureIndex=%d/%d, FPS=%d\n",
                          loopCount, m_currentTextureIndex, m_textureCount, frameInSecond);
                LogToFile(logMsg);
            }
            frameInSecond = 0;
            lastSecondTime = currentTime;
        }
    }

    // 恢复线程优先级
    if (previousPriority != THREAD_PRIORITY_ERROR_RETURN) {
        SetThreadPriority(GetCurrentThread(), previousPriority);
    } else {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    }

    char logMsg[256];
    sprintf_s(logMsg, "Run() exited after %d loops\n", loopCount);
    LogToFile(logMsg);
    PublishTelemetryState(false);
}

void DXOverlay::Stop() {
    m_running = false;
    PublishTelemetryState(false);
}
