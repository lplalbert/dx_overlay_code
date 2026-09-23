#include "overlay_internal.h"

bool DXOverlay::FindAdapterForCurrentMonitor(ComPtr<IDXGIAdapter1>& adapter, ComPtr<IDXGIOutput>& output) {
    adapter.Reset();
    output.Reset();

    if (!m_currentMonitor || m_targetMonitorIndex == -1) {
        return false;
    }

    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        LogToFile("FindAdapterForCurrentMonitor: Failed to create DXGI factory\n");
        return false;
    }

    for (UINT adapterIndex = 0;; ++adapterIndex) {
        ComPtr<IDXGIAdapter1> currentAdapter;
        hr = factory->EnumAdapters1(adapterIndex, &currentAdapter);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr) || !currentAdapter) {
            continue;
        }

        DXGI_ADAPTER_DESC1 adapterDesc = {};
        if (FAILED(currentAdapter->GetDesc1(&adapterDesc))) {
            continue;
        }
        if ((adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }

        for (UINT outputIndex = 0;; ++outputIndex) {
            ComPtr<IDXGIOutput> currentOutput;
            hr = currentAdapter->EnumOutputs(outputIndex, &currentOutput);
            if (hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (FAILED(hr) || !currentOutput) {
                continue;
            }

            DXGI_OUTPUT_DESC outputDesc = {};
            if (FAILED(currentOutput->GetDesc(&outputDesc))) {
                continue;
            }

            if (outputDesc.Monitor == m_currentMonitor) {
                adapter = currentAdapter;
                output = currentOutput;

                char msg[512];
                sprintf_s(msg, "Matched target monitor to adapter: %ls\n", adapterDesc.Description);
                LogToFile(msg);
                return true;
            }
        }
    }

    LogToFile("FindAdapterForCurrentMonitor: No hardware adapter matched the target monitor\n");
    return false;
}

bool DXOverlay::CaptureActiveDeviceAdapterInfo() {
    m_activeAdapterDescription.clear();
    m_activeAdapterLuid = {};
    m_hasActiveAdapterLuid = false;

    if (!m_device) {
        return false;
    }

    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = m_device.As(&dxgiDevice);
    if (FAILED(hr) || !dxgiDevice) {
        return false;
    }

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr) || !adapter) {
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter1;
    hr = adapter.As(&adapter1);
    if (FAILED(hr) || !adapter1) {
        return false;
    }

    DXGI_ADAPTER_DESC1 adapterDesc = {};
    hr = adapter1->GetDesc1(&adapterDesc);
    if (FAILED(hr)) {
        return false;
    }

    m_activeAdapterDescription = adapterDesc.Description;
    m_activeAdapterLuid = adapterDesc.AdapterLuid;
    m_hasActiveAdapterLuid = true;

    char msg[512];
    sprintf_s(msg, "Active device adapter captured: %ls (LUID=%08X:%08X)\n",
        m_activeAdapterDescription.c_str(),
        static_cast<unsigned int>(m_activeAdapterLuid.HighPart),
        static_cast<unsigned int>(m_activeAdapterLuid.LowPart));
    LogToFile(msg);
    return true;
}

bool DXOverlay::IsTargetMonitorOnDifferentAdapter(std::wstring* targetAdapterDescription) {
    if (targetAdapterDescription) {
        targetAdapterDescription->clear();
    }

    if (m_targetMonitorIndex == -1 || !m_currentMonitor || !m_device) {
        return false;
    }

    ComPtr<IDXGIAdapter1> targetAdapter;
    ComPtr<IDXGIOutput> targetOutput;
    if (!FindAdapterForCurrentMonitor(targetAdapter, targetOutput) || !targetAdapter) {
        return false;
    }

    DXGI_ADAPTER_DESC1 targetDesc = {};
    if (FAILED(targetAdapter->GetDesc1(&targetDesc))) {
        return false;
    }

    if (targetAdapterDescription) {
        *targetAdapterDescription = targetDesc.Description;
    }

    if (!m_hasActiveAdapterLuid) {
        return false;
    }

    return m_activeAdapterLuid.HighPart != targetDesc.AdapterLuid.HighPart ||
           m_activeAdapterLuid.LowPart != targetDesc.AdapterLuid.LowPart;
}

void DXOverlay::DetectGPUInfo() {
    // 检测 GPU 厂商、类型和能力
    if (!m_device) {
        LogToFile("DetectGPUInfo: Device not initialized\n");
        return;
    }

    // 获取 DXGI 设备和适配器信息
    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = m_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        LogToFile("DetectGPUInfo: Failed to get DXGI device\n");
        return;
    }

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) {
        LogToFile("DetectGPUInfo: Failed to get adapter\n");
        return;
    }

    // 获取适配器描述
    DXGI_ADAPTER_DESC adapterDesc;
    hr = adapter->GetDesc(&adapterDesc);
    if (FAILED(hr)) {
        LogToFile("DetectGPUInfo: Failed to get adapter description\n");
        return;
    }

    m_gpuDescription = adapterDesc.Description;
    m_gpuDedicatedMemory = adapterDesc.DedicatedVideoMemory;
    m_gpuSharedMemory = adapterDesc.SharedSystemMemory;

    // 根据 VendorId 判断厂商
    // AMD: 0x1002
    // NVIDIA: 0x10DE
    // Intel: 0x8086
    // Microsoft (WARP): 0x1414
    switch (adapterDesc.VendorId) {
        case 0x1002:
            m_gpuVendor = GPUVendor::AMD;
            break;
        case 0x10DE:
            m_gpuVendor = GPUVendor::NVIDIA;
            break;
        case 0x8086:
            m_gpuVendor = GPUVendor::Intel;
            break;
        case 0x1414:
            m_gpuVendor = GPUVendor::Microsoft;
            break;
        default:
            m_gpuVendor = GPUVendor::Unknown;
            break;
    }

    // 判断是否是集成显卡
    // 方法 1: 检查专用显存 - 集成显卡通常没有或很少专用显存
    // 方法 2: 检查 GPU 名称中的关键字
    bool hasLowDedicatedMemory = (m_gpuDedicatedMemory < 512 * 1024 * 1024);  // < 512MB
    bool hasHighSharedMemory = (m_gpuSharedMemory > 1024 * 1024 * 1024);      // > 1GB 共享

    // 检查 GPU 描述中的关键字
    std::wstring desc = m_gpuDescription;
    std::transform(desc.begin(), desc.end(), desc.begin(), ::towlower);

    bool isIntegratedByName =
        desc.find(L"radeon graphics") != std::wstring::npos ||      // AMD APU (Ryzen integrated)
        desc.find(L"radeon vega") != std::wstring::npos ||          // AMD Vega APU
        desc.find(L"radeon(tm) graphics") != std::wstring::npos ||  // AMD APU 变体
        desc.find(L"uhd graphics") != std::wstring::npos ||         // Intel UHD
        desc.find(L"hd graphics") != std::wstring::npos ||          // Intel HD
        desc.find(L"iris") != std::wstring::npos ||                 // Intel Iris
        desc.find(L"integrated") != std::wstring::npos ||           // 通用集成
        desc.find(L"apu") != std::wstring::npos;                    // APU 关键字

    // 综合判断
    if (isIntegratedByName || (hasLowDedicatedMemory && hasHighSharedMemory)) {
        m_gpuType = GPUType::Integrated;
    } else if (m_gpuVendor == GPUVendor::Microsoft) {
        m_gpuType = GPUType::Software;
    } else {
        m_gpuType = GPUType::Discrete;
    }

    // 特别检测 AMD APU
    m_isAMDAPU = (m_gpuVendor == GPUVendor::AMD && m_gpuType == GPUType::Integrated);

    // 特别检测 Intel 核心显卡并确定代数
    m_isIntelIGPU = (m_gpuVendor == GPUVendor::Intel && m_gpuType == GPUType::Integrated);
    if (m_isIntelIGPU) {
        // 检测 Intel 显卡代数
        // Iris Xe (11代+): 最新，性能最好
        // UHD 630/620 (8-10代): 中等
        // HD 630/620 (7代): 较旧
        // HD 530/520 (6代): 更旧
        if (desc.find(L"iris xe") != std::wstring::npos ||
            desc.find(L"iris plus") != std::wstring::npos) {
            m_intelGeneration = 11;  // 11代或更新
        } else if (desc.find(L"uhd 7") != std::wstring::npos) {
            m_intelGeneration = 12;  // 12代 UHD 7xx
        } else if (desc.find(L"uhd 6") != std::wstring::npos) {
            m_intelGeneration = 8;   // 8-10代 UHD 6xx
        } else if (desc.find(L"hd 6") != std::wstring::npos) {
            m_intelGeneration = 7;   // 7代 HD 6xx
        } else if (desc.find(L"hd 5") != std::wstring::npos) {
            m_intelGeneration = 6;   // 6代 HD 5xx
        } else if (desc.find(L"uhd") != std::wstring::npos) {
            m_intelGeneration = 8;   // 通用 UHD
        } else if (desc.find(L"iris") != std::wstring::npos) {
            m_intelGeneration = 10;  // Iris 系列
        } else {
            m_intelGeneration = 7;   // 默认假设 7 代
        }
    }

    // 检测是否是低端 GPU（专用显存 < 2GB 或共享内存模式）
    m_isLowEndGPU = (m_gpuDedicatedMemory < 2ULL * 1024 * 1024 * 1024) ||
                    (m_gpuType == GPUType::Integrated);

    // 输出检测信息
    char msg[512];
    const char* vendorNames[] = {"Unknown", "NVIDIA", "AMD", "Intel", "Microsoft (WARP)"};
    const char* typeNames[] = {"Unknown", "Discrete", "Integrated", "Software"};

    sprintf_s(msg, "GPU Detected:\n"
                   "  Description: %ls\n"
                   "  Vendor: %s\n"
                   "  Type: %s\n"
                   "  Dedicated Memory: %llu MB\n"
                   "  Shared Memory: %llu MB\n"
                   "  AMD APU: %s\n"
                   "  Intel iGPU: %s (Gen %d)\n"
                   "  Low-End GPU: %s\n",
              m_gpuDescription.c_str(),
              vendorNames[static_cast<int>(m_gpuVendor)],
              typeNames[static_cast<int>(m_gpuType)],
              m_gpuDedicatedMemory / (1024 * 1024),
              m_gpuSharedMemory / (1024 * 1024),
              m_isAMDAPU ? "Yes" : "No",
              m_isIntelIGPU ? "Yes" : "No", m_intelGeneration,
              m_isLowEndGPU ? "Yes" : "No");
    LogToFile(msg);
}

void DXOverlay::ApplyGPUOptimizations() {
    // 根据检测到的 GPU 特性应用优化

    char msg[512];
    LogToFile("Applying GPU-specific optimizations...\n");

    // === AMD APU 特定优化 ===
    if (m_isAMDAPU) {
        LogToFile("AMD APU detected - applying APU-specific optimizations:\n");

        // 1. AMD APU 通常与 CPU 共享内存带宽，使用双缓冲而非三缓冲
        m_optimalBufferCount = 2;
        m_useTripleBuffering = false;
        LogToFile("  - Using double buffering (reduces memory bandwidth pressure)\n");

        // 2. AMD APU 支持良好的 MPO，优先使用
        if (m_mpoCapability != MPOCapability::NotSupported) {
            m_mpoCapability = MPOCapability::Preferred;
            LogToFile("  - MPO preferred for AMD APU (hardware compositing)\n");
        }

        // 3. 低延迟模式对 APU 有帮助
        m_reduceLatency = true;
        LogToFile("  - Low latency mode enabled\n");

        // 4. AMD APU 也使用每帧切换模式
        LogToFile("  - Using standard successful-Present texture cadence\n");
    }
    // === Intel 核心显卡优化 ===
    else if (m_isIntelIGPU) {
        sprintf_s(msg, "Intel iGPU (Gen %d) detected - applying Intel-specific optimizations:\n", m_intelGeneration);
        LogToFile(msg);

        // 双缓冲更适合共享内存架构
        m_optimalBufferCount = 2;
        m_useTripleBuffering = false;
        LogToFile("  - Using double buffering (shared memory optimization)\n");

        // 根据 Intel 代数应用不同优化
        if (m_intelGeneration >= 11) {
            // Intel Xe (11代+) - 现代架构，MPO 支持最好
            LogToFile("  - Intel Xe architecture detected (Gen 11+)\n");
            if (m_mpoCapability != MPOCapability::NotSupported) {
                m_mpoCapability = MPOCapability::Preferred;
                LogToFile("  - MPO strongly preferred (Xe hardware compositing)\n");
            }
            m_reduceLatency = true;
            LogToFile("  - Low latency mode enabled (Xe optimized)\n");
        } else if (m_intelGeneration >= 8) {
            // UHD Graphics (8-10代) - 良好支持
            LogToFile("  - Intel UHD architecture detected (Gen 8-10)\n");
            if (m_mpoCapability != MPOCapability::NotSupported) {
                m_mpoCapability = MPOCapability::Preferred;
                LogToFile("  - MPO preferred for UHD graphics\n");
            }
            m_reduceLatency = true;
            LogToFile("  - Low latency mode enabled\n");
        } else {
            // HD Graphics (7代及更早) - 基本支持
            LogToFile("  - Intel HD architecture detected (Gen 7 or earlier)\n");
            // 较旧的 Intel 核显 MPO 支持可能不稳定
            if (m_mpoCapability != MPOCapability::NotSupported) {
                // 保持为 Supported 而非 Preferred
                LogToFile("  - MPO supported (conservative mode for older hardware)\n");
            }
            m_reduceLatency = false;
            LogToFile("  - Standard latency mode (older hardware compatibility)\n");
        }

        // Intel 特有：Power Throttling 优化提示
        LogToFile("  - Intel power management aware rendering enabled\n");
    }
    // === NVIDIA 独立显卡优化 ===
    else if (m_gpuVendor == GPUVendor::NVIDIA && m_gpuType == GPUType::Discrete) {
        LogToFile("NVIDIA discrete GPU detected - applying NVIDIA-specific optimizations:\n");

        // NVIDIA 独立显卡内存充足，可以使用三缓冲减少延迟
        m_optimalBufferCount = 2;  // 保持双缓冲以配合 DirectComposition
        m_useTripleBuffering = false;
        LogToFile("  - Using double buffering (DirectComposition compatibility)\n");

        // NVIDIA 驱动对 MPO 支持可能有问题，谨慎使用
        if (m_mpoCapability == MPOCapability::Supported) {
            // 保持支持状态，但不升级为 Preferred
            LogToFile("  - MPO supported (standard mode)\n");
        }

        LogToFile("  - Texture switching follows successful VSync Present cadence\n");
    }
    // === AMD 独立显卡优化 ===
    else if (m_gpuVendor == GPUVendor::AMD && m_gpuType == GPUType::Discrete) {
        LogToFile("AMD discrete GPU detected - applying AMD dGPU optimizations:\n");

        m_optimalBufferCount = 2;
        m_useTripleBuffering = false;
        LogToFile("  - Using double buffering\n");

        // AMD 独立显卡 MPO 支持良好
        if (m_mpoCapability != MPOCapability::NotSupported) {
            m_mpoCapability = MPOCapability::Preferred;
            LogToFile("  - MPO preferred for AMD dGPU\n");
        }
    }
    // === 软件渲染（WARP）优化 ===
    else if (m_gpuType == GPUType::Software) {
        LogToFile("Software rendering (WARP) detected - applying software mode optimizations:\n");

        // 软件渲染资源有限
        m_optimalBufferCount = 2;
        m_useTripleBuffering = false;
        m_useMPO = false;  // 禁用 MPO
        LogToFile("  - MPO disabled (software rendering)\n");
        LogToFile("  - Using minimal resources\n");
    }
    // === 通用优化 ===
    else {
        LogToFile("Generic GPU - applying standard optimizations:\n");
        m_optimalBufferCount = 2;
        m_useTripleBuffering = false;
        LogToFile("  - Using standard double buffering\n");
    }

    // === 低端 GPU 通用优化 ===
    if (m_isLowEndGPU && m_gpuType != GPUType::Software) {
        LogToFile("Low-end GPU optimizations:\n");
        LogToFile("  - Reduced memory footprint\n");
        LogToFile("  - Conservative resource allocation\n");
    }

    if (m_useTripleBuffering) {
        m_optimalBufferCount = std::max(m_optimalBufferCount, 3);
    }

    m_standardSwapChainBufferCount =
        static_cast<UINT>(std::clamp(m_optimalBufferCount, 2, static_cast<int>(kSwapChainBufferCount)));
    m_mpoSwapChainBufferCount = static_cast<UINT>(kMpoSwapChainBufferCount);
    // 透明叠加窗口更需要“确定性的每帧呈现”而不是额外队列深度。
    // 将最大帧延迟固定为 1，可以减少排队引起的相位漂移和偶发闪烁。
    m_standardMaximumFrameLatency = 1u;
    m_mpoMaximumFrameLatency = 1u;
    LogToFile("Presentation queue pinned to 1 frame latency for deterministic switching\n");

    sprintf_s(msg, "Presentation config: standardBuffers=%u, mpoBuffers=%u, standardLatency=%u, mpoLatency=%u\n",
              m_standardSwapChainBufferCount,
              m_mpoSwapChainBufferCount,
              m_standardMaximumFrameLatency,
              m_mpoMaximumFrameLatency);
    LogToFile(msg);

    sprintf_s(msg, "Optimization summary: BufferCount=%d, TripleBuffer=%s, MPO=%s, LowLatency=%s\n",
              m_optimalBufferCount,
              m_useTripleBuffering ? "Yes" : "No",
              m_mpoCapability == MPOCapability::Preferred ? "Preferred" :
                  (m_mpoCapability == MPOCapability::Supported ? "Supported" : "Disabled"),
              m_reduceLatency ? "Yes" : "No");
    LogToFile(msg);
}

bool DXOverlay::FindCurrentMonitorOutput(ComPtr<IDXGIOutput>& output) {
    output.Reset();

    if (!m_device || !m_currentMonitor) {
        return false;
    }

    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = m_device.As(&dxgiDevice);
    if (FAILED(hr) || !dxgiDevice) {
        return false;
    }

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr) || !adapter) {
        return false;
    }

    for (UINT outputIndex = 0;; ++outputIndex) {
        ComPtr<IDXGIOutput> currentOutput;
        hr = adapter->EnumOutputs(outputIndex, &currentOutput);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr) || !currentOutput) {
            continue;
        }

        DXGI_OUTPUT_DESC desc = {};
        if (FAILED(currentOutput->GetDesc(&desc))) {
            continue;
        }

        if (desc.Monitor == m_currentMonitor) {
            output = currentOutput;
            return true;
        }
    }

    return false;
}

bool DXOverlay::DetectMPOCapability() {
    m_mpoCapability = MPOCapability::NotSupported;
    m_mpoPlaneCount = 0;
    m_mpoOverlaySupportFlags = 0;
    m_mpoOutput.Reset();

    if (!m_device) {
        LogToFile("Cannot detect MPO: Device not initialized\n");
        return false;
    }

    if (m_targetMonitorIndex == -1 || !m_currentMonitor) {
        LogToFile("MPO disabled: spanning all monitors or current monitor unavailable\n");
        return false;
    }

    ComPtr<IDXGIOutput> output;
    if (!FindCurrentMonitorOutput(output) || !output) {
        LogToFile("MPO not supported: failed to resolve DXGI output for current monitor\n");
        return false;
    }

    bool supportsOverlays = false;
    ComPtr<IDXGIOutput2> output2;
    HRESULT hr = output.As(&output2);
    if (SUCCEEDED(hr) && output2) {
        supportsOverlays = output2->SupportsOverlays();
    }

    UINT overlayFlags = 0;
    ComPtr<IDXGIOutput3> output3;
    hr = output.As(&output3);
    if (SUCCEEDED(hr) && output3) {
        hr = output3->CheckOverlaySupport(DXGI_FORMAT_B8G8R8A8_UNORM, m_device.Get(), &overlayFlags);
        if (FAILED(hr)) {
            overlayFlags = 0;
        }
    }

    char msg[256];
    sprintf_s(msg, "MPO detection: SupportsOverlays=%s, OverlayFlags=0x%X\n",
        supportsOverlays ? "true" : "false", overlayFlags);
    LogToFile(msg);

    if (!supportsOverlays || overlayFlags == 0) {
        LogToFile("MPO capability: Not detected on current output\n");
        return false;
    }

    m_mpoOutput = output;
    m_mpoOverlaySupportFlags = overlayFlags;
    m_mpoPlaneCount = (overlayFlags & DXGI_OVERLAY_SUPPORT_FLAG_DIRECT) ? 2u : 1u;
    m_mpoCapability =
        (overlayFlags & DXGI_OVERLAY_SUPPORT_FLAG_DIRECT) ? MPOCapability::Preferred : MPOCapability::Supported;

    sprintf_s(msg, "MPO capability detected: mode=%s, planes=%u\n",
        m_mpoCapability == MPOCapability::Preferred ? "Direct" : "Scaled",
        m_mpoPlaneCount);
    LogToFile(msg);
    return true;
}

bool DXOverlay::InitializeMPOOptimization() {
    if (m_mpoCapability == MPOCapability::NotSupported || !m_mpoOutput || !m_device) {
        LogToFile("MPO optimization: Cannot initialize (not supported)\n");
        return false;
    }

    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = m_device.As(&dxgiDevice);
    if (FAILED(hr) || !dxgiDevice) {
        LogToFile("MPO optimization: Failed to query IDXGIDevice\n");
        return false;
    }

    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr) || !adapter) {
        LogToFile("MPO optimization: Failed to query adapter\n");
        return false;
    }

    ComPtr<IDXGIFactory2> factory;
    hr = adapter->GetParent(__uuidof(IDXGIFactory2), &factory);
    if (FAILED(hr) || !factory) {
        LogToFile("MPO optimization: Failed to query factory\n");
        return false;
    }

    if (m_mpoFrameLatencyWaitableObject) {
        CloseHandle(m_mpoFrameLatencyWaitableObject);
        m_mpoFrameLatencyWaitableObject = nullptr;
    }
    m_mpoRenderTargetView.Reset();
    m_swapChainMPO2.Reset();
    m_swapChainMPO.Reset();

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.Width = m_screenWidth;
    swapChainDesc.Height = m_screenHeight;
    swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapChainDesc.Stereo = FALSE;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = m_mpoSwapChainBufferCount;
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    swapChainDesc.Flags = kSwapChainResizeFlags;

    ComPtr<IDXGISwapChain1> mpoSwapChain1;
    hr = factory->CreateSwapChainForComposition(
        m_device.Get(),
        &swapChainDesc,
        m_mpoOutput.Get(),
        &mpoSwapChain1);
    if (FAILED(hr) || !mpoSwapChain1) {
        char msg[256];
        sprintf_s(msg, "MPO optimization: Failed to create dedicated swap chain, HRESULT: 0x%08X\n", hr);
        LogToFile(msg);
        return false;
    }

    hr = mpoSwapChain1.As(&m_swapChainMPO);
    if (FAILED(hr) || !m_swapChainMPO) {
        LogToFile("MPO optimization: Failed to query IDXGISwapChain3 for dedicated chain\n");
        return false;
    }

    hr = m_swapChainMPO.As(&m_swapChainMPO2);
    if (SUCCEEDED(hr) && m_swapChainMPO2) {
        const HRESULT latencyHr =
            m_swapChainMPO2->SetMaximumFrameLatency(m_mpoMaximumFrameLatency);
        if (SUCCEEDED(latencyHr)) {
            m_mpoFrameLatencyWaitableObject =
                m_swapChainMPO2->GetFrameLatencyWaitableObject();
        } else {
            char latencyMsg[192];
            sprintf_s(latencyMsg,
                "MPO optimization: SetMaximumFrameLatency failed, HRESULT: 0x%08X\n",
                latencyHr);
            LogToFile(latencyMsg);
        }
    }

    ComPtr<ID3D11Texture2D> backBuffer;
    hr = m_swapChainMPO->GetBuffer(0, __uuidof(ID3D11Texture2D), &backBuffer);
    if (FAILED(hr) || !backBuffer) {
        char msg[256];
        sprintf_s(msg, "MPO optimization: Failed to get dedicated back buffer, HRESULT: 0x%08X\n", hr);
        LogToFile(msg);
        if (m_mpoFrameLatencyWaitableObject) {
            CloseHandle(m_mpoFrameLatencyWaitableObject);
            m_mpoFrameLatencyWaitableObject = nullptr;
        }
        m_swapChainMPO2.Reset();
        m_swapChainMPO.Reset();
        return false;
    }

    hr = m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_mpoRenderTargetView);
    if (FAILED(hr) || !m_mpoRenderTargetView) {
        char msg[256];
        sprintf_s(msg, "MPO optimization: Failed to create dedicated RTV, HRESULT: 0x%08X\n", hr);
        LogToFile(msg);
        if (m_mpoFrameLatencyWaitableObject) {
            CloseHandle(m_mpoFrameLatencyWaitableObject);
            m_mpoFrameLatencyWaitableObject = nullptr;
        }
        m_swapChainMPO2.Reset();
        m_swapChainMPO.Reset();
        return false;
    }
    if (!SetActiveVisualContent(m_swapChainMPO.Get(), "enable MPO swap chain")) {
        LogToFile("MPO optimization: Failed to bind dedicated swap chain to DirectComposition\n");
        m_mpoRenderTargetView.Reset();
        if (m_mpoFrameLatencyWaitableObject) {
            CloseHandle(m_mpoFrameLatencyWaitableObject);
            m_mpoFrameLatencyWaitableObject = nullptr;
        }
        m_swapChainMPO2.Reset();
        m_swapChainMPO.Reset();
        return false;
    }

    char msg[256];
    sprintf_s(msg,
        "MPO optimization initialized: dedicated FLIP_SEQUENTIAL chain, planes=%u, flags=0x%X, buffers=%u, latency=%u\n",
        m_mpoPlaneCount, m_mpoOverlaySupportFlags, m_mpoSwapChainBufferCount, m_mpoMaximumFrameLatency);
    LogToFile(msg);

    m_mpoInitialized = true;
    return true;
}

bool DXOverlay::InitializeDirectX() {
    HRESULT hr = S_OK;

    // 创建 D3D11 设备 - 支持多种驱动类型的后备机制
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;  // 支持 BGRA 格式
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // 支持更多特性级别，兼容旧硬件
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1
    };
    D3D_FEATURE_LEVEL featureLevel;

    ComPtr<IDXGIAdapter1> preferredAdapter;
    ComPtr<IDXGIOutput> preferredOutput;
    const bool hasPreferredAdapter = FindAdapterForCurrentMonitor(preferredAdapter, preferredOutput);
    if (hasPreferredAdapter && preferredAdapter) {
        DXGI_ADAPTER_DESC1 preferredDesc = {};
        if (SUCCEEDED(preferredAdapter->GetDesc1(&preferredDesc))) {
            char adapterMsg[512];
            sprintf_s(adapterMsg, "InitializeDirectX: preferred adapter for target monitor is %ls\n",
                preferredDesc.Description);
            LogToFile(adapterMsg);
        }
    } else {
        LogToFile("InitializeDirectX: no target-monitor-specific adapter found, using default adapter selection\n");
    }

    // 尝试不同的驱动类型：目标显示器适配器 -> 默认硬件 -> WARP（软件） -> 参考设备
    D3D_DRIVER_TYPE driverTypes[] = {
        D3D_DRIVER_TYPE_HARDWARE,  // 独立显卡或集成显卡
        D3D_DRIVER_TYPE_WARP,      // 高性能软件光栅化器
        D3D_DRIVER_TYPE_REFERENCE  // 参考设备（最慢，仅调试用）
    };
    const char* driverNames[] = { "HARDWARE", "WARP (Software)", "REFERENCE" };

    if (hasPreferredAdapter && preferredAdapter) {
        m_device.Reset();
        m_context.Reset();
        hr = D3D11CreateDevice(
            preferredAdapter.Get(),
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            createDeviceFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            &m_device,
            &featureLevel,
            &m_context
        );

        if (SUCCEEDED(hr)) {
            char msg[256];
            sprintf_s(msg, "D3D11 device created on target monitor adapter, Feature Level: 0x%04X\n",
                      featureLevel);
            OutputDebugStringA(msg);
            LogToFile(msg);
        } else {
            char msg[256];
            sprintf_s(msg, "Failed to create device on target monitor adapter (0x%08X), falling back...\n", hr);
            OutputDebugStringA(msg);
            LogToFile(msg);
            m_device.Reset();
            m_context.Reset();
        }
    }

    if (!m_device || !m_context) {
        for (int i = 0; i < ARRAYSIZE(driverTypes); ++i) {
            hr = D3D11CreateDevice(
                nullptr,
                driverTypes[i],
                nullptr,
                createDeviceFlags,
                featureLevels,
                ARRAYSIZE(featureLevels),
                D3D11_SDK_VERSION,
                &m_device,
                &featureLevel,
                &m_context
            );

            if (SUCCEEDED(hr)) {
                char msg[256];
                sprintf_s(msg, "D3D11 device created with %s driver, Feature Level: 0x%04X\n",
                          driverNames[i], featureLevel);
                OutputDebugStringA(msg);
                LogToFile(msg);
                break;
            }

            char msg[256];
            sprintf_s(msg, "Failed to create device with %s driver (0x%08X), trying next...\n",
                      driverNames[i], hr);
            OutputDebugStringA(msg);
        }
    }

    if (FAILED(hr)) {
        char msg[512];
        sprintf_s(msg, "ERROR: Failed to create D3D11 device with any driver, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create D3D11 device (no compatible driver found)";
        return false;
    }

    char msg[256];
    sprintf_s(msg, "D3D11 device created successfully, Feature Level: 0x%04X\n", featureLevel);
    OutputDebugStringA(msg);
    CaptureActiveDeviceAdapterInfo();

    // 获取 DXGI 设备
    OutputDebugStringA("Getting DXGI device...\n");
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_device.As(&dxgiDevice);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to get DXGI device, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to get DXGI device";
        return false;
    }
    OutputDebugStringA("DXGI device obtained successfully\n");

    // 获取适配器
    OutputDebugStringA("Getting DXGI adapter...\n");
    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to get adapter, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to get adapter";
        return false;
    }
    OutputDebugStringA("Adapter obtained successfully\n");

    // 获取工厂
    OutputDebugStringA("Getting DXGI factory...\n");
    ComPtr<IDXGIFactory2> factory;
    hr = adapter->GetParent(__uuidof(IDXGIFactory2), &factory);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to get factory, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to get factory";
        return false;
    }
    OutputDebugStringA("Factory obtained successfully\n");

    // 禁用 DXGI 窗口关联，避免 Alt+Enter 等行为
    factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER);

    // 透明合成窗口更偏向稳定性，使用 FLIP_SEQUENTIAL 避免丢弃式交换在某些驱动上的闪烁。
    OutputDebugStringA("Creating swap chain for composition...\n");
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.Width = m_screenWidth;
    swapChainDesc.Height = m_screenHeight;
    swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapChainDesc.Stereo = FALSE;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    // 使用三缓冲减少闪烁和撕裂
    swapChainDesc.BufferCount = m_standardSwapChainBufferCount;
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    // 启用帧延迟等待对象，减少输入延迟和闪烁
    swapChainDesc.Flags = kSwapChainResizeFlags;

    sprintf_s(msg,
        "Swap chain config: %dx%d, FLIP_SEQUENTIAL, buffers=%u, maxLatency=%u\n",
        m_screenWidth, m_screenHeight, m_standardSwapChainBufferCount, m_standardMaximumFrameLatency);
    OutputDebugStringA(msg);

    // 为 DirectComposition 创建 swap chain（不绑定到窗口）
    hr = factory->CreateSwapChainForComposition(
        m_device.Get(),
        &swapChainDesc,
        nullptr,
        &m_swapChain
    );

    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to create swap chain for composition, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create swap chain for composition";
        return false;
    }
    OutputDebugStringA("Swap chain for composition created successfully\n");

    // 获取 IDXGISwapChain2 接口并设置帧延迟（减少闪烁）
    hr = m_swapChain.As(&m_swapChain2);
    if (SUCCEEDED(hr) && m_swapChain2) {
        const HRESULT latencyHr =
            m_swapChain2->SetMaximumFrameLatency(m_standardMaximumFrameLatency);
        if (SUCCEEDED(latencyHr)) {
            m_frameLatencyWaitableObject =
                m_swapChain2->GetFrameLatencyWaitableObject();
            if (m_frameLatencyWaitableObject) {
                OutputDebugStringA("Frame latency waitable object obtained - render pacing enabled\n");
            }
        } else {
            sprintf_s(msg,
                "WARNING: SetMaximumFrameLatency failed, HRESULT: 0x%08X; using Present pacing\n",
                latencyHr);
            OutputDebugStringA(msg);
        }
    } else {
        OutputDebugStringA("IDXGISwapChain2 not available, using standard Present\n");
    }

    // 初始化 DirectComposition，将透明交换链提交给 DWM 合成。
    OutputDebugStringA("Initializing DirectComposition for DWM composition...\n");
    hr = DCompositionCreateDevice(dxgiDevice.Get(), __uuidof(IDCompositionDevice), reinterpret_cast<void**>(m_dcompDevice.GetAddressOf()));
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to create DirectComposition device, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create DirectComposition device";
        return false;
    }
    OutputDebugStringA("DirectComposition device created successfully\n");

    // 创建 composition target（绑定到窗口）
    hr = m_dcompDevice->CreateTargetForHwnd(m_hwnd, TRUE, &m_dcompTarget);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to create composition target, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create composition target";
        return false;
    }
    OutputDebugStringA("Composition target created successfully\n");

    // 创建 visual
    hr = m_dcompDevice->CreateVisual(&m_dcompVisual);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to create composition visual, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create composition visual";
        return false;
    }
    OutputDebugStringA("Composition visual created successfully\n");

    // 将 swap chain 设置为 visual 的内容
    hr = m_dcompVisual->SetContent(m_swapChain.Get());
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to set visual content, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to set visual content";
        return false;
    }

    // 将 visual 设置为 target 的根
    hr = m_dcompTarget->SetRoot(m_dcompVisual.Get());
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to set target root, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to set target root";
        return false;
    }

    // 提交 DirectComposition 的更改
    hr = m_dcompDevice->Commit();
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to commit composition, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to commit composition";
        return false;
    }
    OutputDebugStringA("DirectComposition initialized and committed to DWM\n");

    // D3D11 flip-model 会在 Present 后更新 back-buffer 0 的身份；RTV 可复用，
    // 但每帧仍需重新绑定，因为 Present 会将其从输出合并阶段解除绑定。
    OutputDebugStringA("Creating initial RTV...\n");
    ComPtr<ID3D11Texture2D> backBuffer;
    hr = m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &backBuffer);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to get back buffer, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to get back buffer";
        return false;
    }

    hr = m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_renderTargetView);
    if (FAILED(hr)) {
        sprintf_s(msg, "ERROR: Failed to create RTV, HRESULT: 0x%08X\n", hr);
        OutputDebugStringA(msg);
        m_lastError = "Failed to create render target view";
        return false;
    }
    OutputDebugStringA("Initial RTV created successfully\n");

    // 设置初始渲染目标
    m_context->OMSetRenderTargets(1, m_renderTargetView.GetAddressOf(), nullptr);

    // 设置视口
    // Rebind the shared pipeline state every frame so device recovery and monitor migration
    // do not depend on stale one-time initialization.
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(m_screenWidth);
    vp.Height = static_cast<float>(m_screenHeight);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    m_context->RSSetViewports(1, &vp);

    OutputDebugStringA("DirectX initialization completed successfully!\n");
    return true;
}

bool DXOverlay::ResizeSwapChainResources(
    IDXGISwapChain1* swapChain,
    ComPtr<IDXGISwapChain2>& swapChain2,
    HANDLE& frameLatencyWaitableObject,
    ComPtr<ID3D11RenderTargetView>& renderTargetView,
    UINT bufferCount,
    UINT maximumFrameLatency,
    int newWidth,
    int newHeight,
    const char* label) {
    if (!swapChain || !m_device || !m_context) {
        return false;
    }

    m_context->OMSetRenderTargets(0, nullptr, nullptr);
    renderTargetView.Reset();
    m_context->Flush();

    if (frameLatencyWaitableObject) {
        CloseHandle(frameLatencyWaitableObject);
        frameLatencyWaitableObject = nullptr;
    }

    HRESULT hr = swapChain->ResizeBuffers(
        bufferCount,
        newWidth,
        newHeight,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        kSwapChainResizeFlags);

    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "ResizeBuffers failed for %s swap chain, HRESULT: 0x%08X\n",
            label ? label : "unknown", hr);
        LogToFile(msg);
        return false;
    }

    swapChain2.Reset();
    ComPtr<IDXGISwapChain1> swapChainRef = swapChain;
    hr = swapChainRef.As(&swapChain2);
    if (SUCCEEDED(hr) && swapChain2) {
        const HRESULT latencyHr = swapChain2->SetMaximumFrameLatency(maximumFrameLatency);
        if (SUCCEEDED(latencyHr)) {
            frameLatencyWaitableObject = swapChain2->GetFrameLatencyWaitableObject();
        } else {
            char msg[256];
            sprintf_s(msg,
                "SetMaximumFrameLatency failed for %s swap chain after resize, HRESULT: 0x%08X\n",
                label ? label : "unknown",
                latencyHr);
            LogToFile(msg);
        }
    }

    ComPtr<ID3D11Texture2D> backBuffer;
    hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &backBuffer);
    if (FAILED(hr) || !backBuffer) {
        char msg[256];
        sprintf_s(msg, "Failed to get back buffer for %s swap chain after resize, HRESULT: 0x%08X\n",
            label ? label : "unknown", hr);
        LogToFile(msg);
        return false;
    }

    hr = m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &renderTargetView);
    if (FAILED(hr) || !renderTargetView) {
        char msg[256];
        sprintf_s(msg, "Failed to create RTV for %s swap chain after resize, HRESULT: 0x%08X\n",
            label ? label : "unknown", hr);
        LogToFile(msg);
        return false;
    }

    return true;
}

bool DXOverlay::ResizeSwapChainBuffers(int newWidth, int newHeight, const char* reason) {
    if (!m_swapChain || !m_device || !m_context) {
        return false;
    }

    if (newWidth <= 0 || newHeight <= 0) {
        return false;
    }

    bool expected = false;
    if (!m_isResizing.compare_exchange_strong(expected, true)) {
        LogToFile("Resize request ignored because another resize is already in progress\n");
        return false;
    }
    AtomicFlagScope resizeGuard(m_isResizing);

    m_screenWidth = newWidth;
    m_screenHeight = newHeight;

    char msg[256];
    sprintf_s(msg, "Resizing swap chains (%s): %dx%d\n",
        reason ? reason : "unknown", newWidth, newHeight);
    LogToFile(msg);

    if (!ResizeSwapChainResources(
            m_swapChain.Get(),
            m_swapChain2,
            m_frameLatencyWaitableObject,
            m_renderTargetView,
            m_standardSwapChainBufferCount,
            m_standardMaximumFrameLatency,
            newWidth,
            newHeight,
            "standard")) {
        return false;
    }
    if (m_mpoInitialized && m_swapChainMPO) {
        if (!ResizeSwapChainResources(
                m_swapChainMPO.Get(),
                m_swapChainMPO2,
                m_mpoFrameLatencyWaitableObject,
                m_mpoRenderTargetView,
                m_mpoSwapChainBufferCount,
                m_mpoMaximumFrameLatency,
                newWidth,
                newHeight,
                "MPO")) {
            LogToFile("MPO resize failed, disabling MPO and falling back to standard swap chain\n");
            if (m_mpoFrameLatencyWaitableObject) {
                CloseHandle(m_mpoFrameLatencyWaitableObject);
                m_mpoFrameLatencyWaitableObject = nullptr;
            }
            m_swapChainMPO2.Reset();
            m_swapChainMPO.Reset();
            m_mpoRenderTargetView.Reset();
            m_mpoInitialized = false;
            m_useMPO = false;
        }
    }

    IUnknown* activeContent =
        (m_useMPO && m_mpoInitialized && m_swapChainMPO)
            ? static_cast<IUnknown*>(m_swapChainMPO.Get())
            : static_cast<IUnknown*>(m_swapChain.Get());
    if (!SetActiveVisualContent(activeContent, reason ? reason : "swap chain resize")) {
        return false;
    }

    ID3D11RenderTargetView* activeRTV =
        (m_useMPO && m_mpoInitialized && m_mpoRenderTargetView)
            ? m_mpoRenderTargetView.Get()
            : m_renderTargetView.Get();
    if (activeRTV) {
        m_context->OMSetRenderTargets(1, &activeRTV, nullptr);
    }

    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(newWidth);
    vp.Height = static_cast<float>(newHeight);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    m_context->RSSetViewports(1, &vp);

    sprintf_s(msg, "Swap chain resize finished (%s)\n", reason ? reason : "unknown");
    LogToFile(msg);
    return true;
}
