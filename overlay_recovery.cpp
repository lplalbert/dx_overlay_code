#include "overlay_internal.h"

void DXOverlay::RegisterRecoveryFailure(const char* reason, bool exitImmediately) {
    m_consecutiveRecoveryFailures++;

    char msg[512];
    sprintf_s(
        msg,
        "Recovery failure count=%d/%d (%s)\n",
        m_consecutiveRecoveryFailures,
        m_maxConsecutiveRecoveryFailures,
        reason ? reason : "unknown");
    LogToFile(msg);

    if (exitImmediately || m_consecutiveRecoveryFailures >= m_maxConsecutiveRecoveryFailures) {
        char userMessage[768];
        sprintf_s(
            userMessage,
            "The overlay could not recover its rendering device and will now exit.\n\nReason: %s",
            reason ? reason : "Unknown recovery failure");
        TriggerFatalShutdown(reason, userMessage);
    }
}

bool DXOverlay::SetActiveVisualContent(IUnknown* content, const char* reason) {
    if (!m_dcompVisual || !m_dcompDevice || !content) {
        return HandleVisualContentFailure(E_POINTER, "validate DirectComposition content", reason);
    }

    HRESULT hr = m_dcompVisual->SetContent(content);
    if (FAILED(hr)) {
        return HandleVisualContentFailure(hr, "set DirectComposition content", reason);
    }

    hr = m_dcompDevice->Commit();
    if (FAILED(hr)) {
        return HandleVisualContentFailure(hr, "commit DirectComposition content", reason);
    }

    return true;
}

void DXOverlay::QueueRecovery(RecoveryRequest request, HRESULT hr, const char* reason) {
    const int requestedLevel = static_cast<int>(request);
    int currentLevel = m_pendingRecoveryRequest.load();
    while (currentLevel < requestedLevel &&
           !m_pendingRecoveryRequest.compare_exchange_weak(currentLevel, requestedLevel)) {
    }
    m_pendingRecoveryHr.store(static_cast<long>(hr));

    char msg[384];
    sprintf_s(
        msg,
        "Queued %s recovery (%s), HRESULT: 0x%08X\n",
        request == RecoveryRequest::Device ? "device" : "presentation",
        reason ? reason : "unknown",
        static_cast<unsigned int>(hr));
    LogToFile(msg);
}

bool DXOverlay::ProcessQueuedRecovery() {
    const RecoveryRequest request = static_cast<RecoveryRequest>(
        m_pendingRecoveryRequest.exchange(static_cast<int>(RecoveryRequest::None)));
    if (request == RecoveryRequest::None) {
        return true;
    }

    const HRESULT hr = static_cast<HRESULT>(m_pendingRecoveryHr.exchange(static_cast<long>(S_OK)));
    char msg[256];
    sprintf_s(
        msg,
        "Processing queued %s recovery, trigger HRESULT: 0x%08X\n",
        request == RecoveryRequest::Device ? "device" : "presentation",
        static_cast<unsigned int>(hr));
    LogToFile(msg);

    const bool recovered = request == RecoveryRequest::Device
        ? RecoverFromDeviceLoss()
        : RecoverPresentationPipeline("queued runtime recovery");

    if (!recovered && m_running && !m_fatalShutdownTriggered.load()) {
        // 下一轮升级为完整设备恢复，由恢复失败计数器限制重试。
        QueueRecovery(RecoveryRequest::Device, hr, "previous recovery attempt failed");
    }
    return recovered;
}

bool DXOverlay::HandleVisualContentFailure(HRESULT hr, const char* stage, const char* reason) {
    char msg[384];
    sprintf_s(
        msg,
        "Failed to %s (%s), HRESULT: 0x%08X\n",
        stage ? stage : "use DirectComposition content",
        reason ? reason : "unknown",
        hr);
    LogToFile(msg);

    if (!m_running) {
        m_lastError = "DirectComposition visual content initialization failed";
        return false;
    }

    if (m_isRecovering.load()) {
        m_lastError = "DirectComposition visual restore failed during recovery";
        return false;
    }

    QueueRecovery(RecoveryRequest::Presentation, hr, "DirectComposition content failure");
    return false;
}

void DXOverlay::ReleaseAllResources() {
    m_displayPhaseAnchorValid = false;
    m_renderTargetView.Reset();
    m_mpoRenderTargetView.Reset();
    m_textureArraySRV.Reset();
    m_textureArray.Reset();
    m_blendState.Reset();
    m_pointSamplerState.Reset();
    m_constantBuffer.Reset();
    m_vertexBuffer.Reset();
    m_inputLayout.Reset();
    m_pixelShader.Reset();
    m_vertexShader.Reset();

    if (m_dcompVisual) m_dcompVisual.Reset();
    if (m_dcompTarget) m_dcompTarget.Reset();
    if (m_dcompDevice) m_dcompDevice.Reset();

    if (m_frameLatencyWaitableObject) {
        CloseHandle(m_frameLatencyWaitableObject);
        m_frameLatencyWaitableObject = nullptr;
    }
    if (m_mpoFrameLatencyWaitableObject) {
        CloseHandle(m_mpoFrameLatencyWaitableObject);
        m_mpoFrameLatencyWaitableObject = nullptr;
    }

    m_swapChain2.Reset();
    m_swapChainMPO2.Reset();
    m_swapChainMPO.Reset();
    m_mpoOutput.Reset();
    m_mpoOverlaySupportFlags = 0;
    m_swapChain.Reset();

    if (m_context) {
        m_context->ClearState();
        m_context->Flush();
        m_context.Reset();
    }
    m_device.Reset();

    m_useMPO = false;
    m_mpoInitialized = false;
    m_mpoCapability = MPOCapability::NotSupported;
    m_mpoPlaneCount = 0;
    m_activeAdapterDescription.clear();
    m_activeAdapterLuid = {};
    m_hasActiveAdapterLuid = false;
}

bool DXOverlay::ReinitializeDevice() {
    auto failAndRelease = [this](const char* message) {
        LogToFile(message);
        ReleaseAllResources();
        return false;
    };

    LogToFile("ReinitializeDevice: Initializing DirectX on current monitor/adapter...\n");
    if (!InitializeDirectX()) {
        return failAndRelease("ERROR: ReinitializeDevice failed during InitializeDirectX\n");
    }

    LogToFile("ReinitializeDevice: Detecting GPU information...\n");
    DetectGPUInfo();

    m_mpoCapability = MPOCapability::NotSupported;
    m_useMPO = false;
    m_mpoInitialized = false;
    LogToFile("ReinitializeDevice: MPO remains disabled by stability policy\n");

    LogToFile("ReinitializeDevice: Applying GPU-specific optimizations...\n");
    ApplyGPUOptimizations();

    if (m_standardSwapChainBufferCount != kSwapChainBufferCount || m_standardMaximumFrameLatency != 1u) {
        LogToFile("ReinitializeDevice: Applying GPU-specific presentation settings...\n");
        if (!ResizeSwapChainBuffers(m_screenWidth, m_screenHeight, "reinitialize gpu-specific presentation config")) {
            return failAndRelease("ERROR: ReinitializeDevice failed while applying presentation settings\n");
        }
    }

    LogToFile("ReinitializeDevice: Recreating shaders...\n");
    if (!CreateShaders()) {
        return failAndRelease("ERROR: ReinitializeDevice failed while recreating shaders\n");
    }

    if (!m_imagePaths.empty()) {
        LogToFile("ReinitializeDevice: Reloading textures...\n");
        if (!LoadTextures(m_imagePaths)) {
            return failAndRelease("ERROR: ReinitializeDevice failed while reloading textures\n");
        }
    }

    UpdateConstantBuffer();

    m_consecutiveRecoveryFailures = 0;
    m_consecutivePresentFailures = 0;
    m_consecutiveMapFailures = 0;
    m_pendingRecoveryRequest.store(static_cast<int>(RecoveryRequest::None));
    m_pendingRecoveryHr.store(static_cast<long>(S_OK));
    PublishTelemetryState(m_running.load());
    LogToFile("ReinitializeDevice: Completed successfully\n");
    return true;
}

bool DXOverlay::RebuildAllRenderingResources(const char* reason) {
    char msg[384];
    sprintf_s(msg, "Rebuilding all rendering resources (%s)\n", reason ? reason : "unknown");
    LogToFile(msg);

    ReleaseAllResources();
    const DWORD backoffMs = static_cast<DWORD>(
        std::min(250, 50 + std::max(0, m_deviceRecoveryAttempts - 1) * 50));
    Sleep(backoffMs);

    if (!ReinitializeDevice()) {
        LogToFile("ERROR: Full rendering resource rebuild failed\n");
        return false;
    }

    LogToFile("Full rendering resource rebuild completed successfully\n");
    return true;
}

bool DXOverlay::RecoverPresentationPipeline(const char* reason) {
    LogToFile("Attempting presentation-pipeline recovery...\n");

    bool expected = false;
    if (!m_isRecovering.compare_exchange_strong(expected, true)) {
        LogToFile("Presentation recovery skipped because another recovery is in progress\n");
        return false;
    }
    AtomicFlagScope recoveryGuard(m_isRecovering);
    m_isSuspended.store(true);

    struct SuspendScope {
        std::atomic<bool>& suspended;
        explicit SuspendScope(std::atomic<bool>& value) : suspended(value) {}
        ~SuspendScope() { suspended.store(false); }
    } suspendGuard(m_isSuspended);

    // 先尝试低成本的 swap-chain/DirectComposition 重绑定。
    if (m_device && m_context && m_swapChain && m_device->GetDeviceRemovedReason() == S_OK) {
        if (ResizeSwapChainBuffers(
                std::max(1, m_screenWidth),
                std::max(1, m_screenHeight),
                reason ? reason : "presentation recovery")) {
            m_consecutivePresentFailures = 0;
            m_consecutiveMapFailures = 0;
            m_consecutiveRecoveryFailures = 0;
            m_deviceRecoveryAttempts = 0;
            m_pendingRecoveryRequest.store(static_cast<int>(RecoveryRequest::None));
            LogToFile("Presentation pipeline recovered without recreating the D3D device\n");
            return true;
        }
        LogToFile("Lightweight presentation recovery failed; escalating to full resource rebuild\n");
    }

    m_deviceRecoveryAttempts++;
    if (m_deviceRecoveryAttempts > MAX_RECOVERY_ATTEMPTS) {
        RegisterRecoveryFailure("Maximum presentation recovery attempts exceeded", true);
        return false;
    }

    if (!RebuildAllRenderingResources(reason ? reason : "presentation recovery escalation")) {
        RegisterRecoveryFailure("Failed to rebuild resources after presentation failure", false);
        return false;
    }

    m_deviceRecoveryAttempts = 0;
    m_consecutiveRecoveryFailures = 0;
    m_consecutivePresentFailures = 0;
    return true;
}

bool DXOverlay::RecoverFromDeviceLoss() {
    LogToFile("Attempting full recovery from device loss...\n");

    bool expected = false;
    if (!m_isRecovering.compare_exchange_strong(expected, true)) {
        LogToFile("Device recovery is already in progress\n");
        return false;
    }
    AtomicFlagScope recoveryGuard(m_isRecovering);
    m_isSuspended.store(true);

    struct SuspendScope {
        std::atomic<bool>& suspended;
        explicit SuspendScope(std::atomic<bool>& value) : suspended(value) {}
        ~SuspendScope() { suspended.store(false); }
    } suspendGuard(m_isSuspended);

    const HRESULT hrRemoved = m_device ? m_device->GetDeviceRemovedReason() : E_POINTER;
    char msg[256];
    sprintf_s(msg, "Device recovery trigger reason: 0x%08X\n", static_cast<unsigned int>(hrRemoved));
    LogToFile(msg);

    m_deviceRecoveryAttempts++;
    if (m_deviceRecoveryAttempts > MAX_RECOVERY_ATTEMPTS) {
        RegisterRecoveryFailure("Maximum device recovery attempts exceeded", true);
        return false;
    }

    sprintf_s(msg, "Recovery attempt %d/%d\n", m_deviceRecoveryAttempts, MAX_RECOVERY_ATTEMPTS);
    LogToFile(msg);

    if (!RebuildAllRenderingResources("device loss")) {
        RegisterRecoveryFailure("Failed to rebuild device resources", false);
        return false;
    }

    m_deviceRecoveryAttempts = 0;
    m_consecutiveRecoveryFailures = 0;
    m_consecutivePresentFailures = 0;
    m_consecutiveMapFailures = 0;
    LogToFile("Device recovery completed successfully\n");
    return true;
}
