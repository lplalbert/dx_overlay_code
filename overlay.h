#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>  // For IDXGISwapChain2
#include <dxgi1_4.h>  // For DXGI 1.4 and HDR support
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <wincodec.h>
#include <dcomp.h>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdint>
#include "overlay_telemetry.h"

using Microsoft::WRL::ComPtr;

// 全局日志开关（在 overlay.cpp 中定义）
extern bool g_enableLogging;
void InitializeLogging(const std::wstring& exeDir, bool enableVerboseLogging);
void WriteStartupLog(const char* msg);
void CompleteStartupLogging();

// MPO (Multiple Plane Overlay) 支持枚举
enum class MPOCapability {
    NotSupported,        // 不支持 MPO
    Supported,           // 支持 MPO
    Preferred            // 首选使用 MPO（性能最优）
};

// GPU 厂商枚举
enum class GPUVendor {
    Unknown,
    NVIDIA,
    AMD,
    Intel,
    Microsoft  // WARP 软件渲染
};

// GPU 类型枚举
enum class GPUType {
    Unknown,
    Discrete,    // 独立显卡
    Integrated,  // 集成显卡/APU
    Software     // 软件渲染（WARP）
};

struct Vertex {
    float x, y;
    float u, v;
};

class DXOverlay {
public:
    DXOverlay();
    ~DXOverlay();

    // 初始化
    bool Initialize(
        const std::vector<std::wstring>& imagePaths,
        float staticAlpha = 0.04f,
        float dynamicAlpha = 0.04f);

    // 设置目标显示器（在 Initialize 之前调用）
    // monitorIndex: 0 = 主显示器, 1+ = 其他显示器, -1 = 覆盖所有显示器
    void SetTargetMonitor(int monitorIndex) { m_targetMonitorIndex = monitorIndex; }
    void SetSwitchingBehavior(int holdFrames, int transitionFrames);
    int GetMonitorCount();  // 获取显示器数量

    // 运行主循环
    void Run();

    // 停止
    void Stop();

    // 获取最后的错误信息
    const std::string& GetLastError() const { return m_lastError; }

private:
    // 窗口相关
    bool CreateOverlayWindow();
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // GPU 检测和优化
    void DetectGPUInfo();           // 检测 GPU 信息（厂商、类型）
    void ApplyGPUOptimizations();   // 根据 GPU 特性应用优化
    bool FindAdapterForCurrentMonitor(ComPtr<IDXGIAdapter1>& adapter, ComPtr<IDXGIOutput>& output);
    bool CaptureActiveDeviceAdapterInfo();
    bool IsTargetMonitorOnDifferentAdapter(std::wstring* targetAdapterDescription = nullptr);

    // MPO (Multiple Plane Overlay) 检测和优化
    bool DetectMPOCapability();
    bool InitializeMPOOptimization();
    void RenderWithMPO();      // MPO 优化的渲染路径
    void RenderWithoutMPO();   // 标准渲染路径
    bool FindCurrentMonitorOutput(ComPtr<IDXGIOutput>& output);
    bool SetActiveVisualContent(IUnknown* content, const char* reason);

    // DirectX 初始化
    bool InitializeDirectX();
    bool CreateShaders();
    bool LoadTextures(const std::vector<std::wstring>& imagePaths);
    bool LoadTextureFromFile(
        const std::wstring& path,
        ComPtr<ID3D11ShaderResourceView>& srv,
        int& width,
        int& height,
        bool* hasNonOpaqueAlpha = nullptr,
        bool* isChannelEncoded = nullptr);
    bool ResizeSwapChainBuffers(int newWidth, int newHeight, const char* reason);
    bool ResizeSwapChainResources(
        IDXGISwapChain1* swapChain,
        ComPtr<IDXGISwapChain2>& swapChain2,
        HANDLE& frameLatencyWaitableObject,
        ComPtr<ID3D11RenderTargetView>& renderTargetView,
        UINT bufferCount,
        UINT maximumFrameLatency,
        int newWidth,
        int newHeight,
        const char* label);

    // 渲染
    void Render();
    void UpdateConstantBuffer();
    bool RenderToSwapChain(
        IDXGISwapChain1* swapChain,
        ComPtr<ID3D11RenderTargetView>& renderTargetView,
        bool useRestrictedPresent,
        IDXGIOutput* restrictOutput,
        const char* label);
    HANDLE GetActiveFrameLatencyWaitableObject() const;

    // 窗口事件处理
    void OnWindowSizeChanged();
    void OnDisplayChange();

    // 多显示器支持
    void EnumerateMonitors();                  // 枚举所有显示器
    bool SelectMonitor(int monitorIndex);     // 选择指定显示器
    static BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);

    // 设备恢复
    enum class RecoveryRequest : int {
        None = 0,
        Presentation = 1,
        Device = 2
    };

    void QueueRecovery(RecoveryRequest request, HRESULT hr, const char* reason);
    bool ProcessQueuedRecovery();
    bool RecoverPresentationPipeline(const char* reason);
    bool RecoverFromDeviceLoss();
    bool RebuildAllRenderingResources(const char* reason);
    void ReleaseAllResources();           // 释放所有 DirectX 资源
    bool ReinitializeDevice();            // 重新初始化设备

    struct TexturePhase {
        int currentTextureIndex = 0;
        int previousTextureIndex = 0;
        float transitionBlend = 1.0f;
    };
    TexturePhase CalculateTexturePhase(std::uint64_t successfulPresentOrdinal) const;
    bool HandleVisualContentFailure(HRESULT hr, const char* stage, const char* reason);
    void TriggerFatalShutdown(const char* logReason, const char* userMessage);
    void RegisterRecoveryFailure(const char* reason, bool exitImmediately);

    // 置顶管理
    void CheckAndRestoreTopmost();        // 检查并恢复置顶状态
    void OnZOrderChanged();               // Z-order 改变时的处理

    // 诊断遥测（供 test_alpha/test_frame3 跨进程读取）
    bool InitializeTelemetry();
    void ShutdownTelemetry();
    void PublishTelemetryState(bool running);
    void PublishPresentResult(
        HRESULT result,
        IDXGISwapChain1* swapChain,
        int submittedTextureIndex,
        std::uint64_t submittedPhaseOrdinal);
    void PublishFrameStatistics(IDXGISwapChain1* swapChain);

    // 窗口
    HWND m_hwnd = nullptr;
    HINSTANCE m_hInstance = nullptr;
    int m_screenWidth = 0;
    int m_screenHeight = 0;
    int m_imageWidth = 0;
    int m_imageHeight = 0;
    int m_windowX = 0;
    int m_windowY = 0;

    // 多显示器支持
    struct MonitorInfo {
        HMONITOR hMonitor;
        RECT rect;           // 显示器区域
        int width;
        int height;
        int refreshRate;
        bool isPrimary;
        std::wstring name;
    };
    std::vector<MonitorInfo> m_monitors;  // 所有显示器列表
    int m_targetMonitorIndex = 0;          // 目标显示器索引 (0=主显示器, -1=所有显示器)
    HMONITOR m_currentMonitor = nullptr;   // 当前使用的显示器句柄

    // DirectX 设备
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGISwapChain1> m_swapChain;
    ComPtr<ID3D11RenderTargetView> m_renderTargetView;

    // 帧延迟等待对象（减少闪烁）
    HANDLE m_frameLatencyWaitableObject = nullptr;
    ComPtr<IDXGISwapChain2> m_swapChain2;          // 用于帧延迟控制

    // 着色器
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11InputLayout> m_inputLayout;

    // 缓冲区
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11Buffer> m_constantBuffer;

    // 纹理管理 - 使用纹理数组优化
    ComPtr<ID3D11ShaderResourceView> m_textureArraySRV;      // 纹理数组 SRV（包含所有图片）
    ComPtr<ID3D11Texture2D> m_textureArray;                   // 纹理数组资源
    int m_textureCount = 0;                                   // 纹理数量
    std::vector<std::wstring> m_imagePaths;                   // 图像路径缓存（用于恢复）
    bool m_dynamicTexturesUseSourceAlpha = true;             // 旧模板是否包含真实 alpha
    bool m_channelEncodedTemplates = false;                  // R=Y, G=Cr(dynamic), B=Cb(static), A=254
    ComPtr<ID3D11SamplerState> m_pointSamplerState;
    ComPtr<ID3D11BlendState> m_blendState;

    // DirectComposition
    ComPtr<IDCompositionDevice> m_dcompDevice;
    ComPtr<IDCompositionTarget> m_dcompTarget;
    ComPtr<IDCompositionVisual> m_dcompVisual;

    // MPO (Multiple Plane Overlay) 支持
    MPOCapability m_mpoCapability = MPOCapability::NotSupported;
    bool m_useMPO = false;               // 是否启用 MPO 优化
    bool m_mpoInitialized = false;       // MPO 初始化成功标志
    ComPtr<IDXGISwapChain3> m_swapChainMPO;  // MPO 模式下的独立 swap chain
    ComPtr<IDXGISwapChain2> m_swapChainMPO2;
    ComPtr<ID3D11RenderTargetView> m_mpoRenderTargetView;
    HANDLE m_mpoFrameLatencyWaitableObject = nullptr;
    ComPtr<IDXGIOutput> m_mpoOutput;
    UINT m_mpoOverlaySupportFlags = 0;
    UINT m_mpoPlaneCount = 0;           // GPU 支持的最大平面数

    // GPU 信息
    GPUVendor m_gpuVendor = GPUVendor::Unknown;
    GPUType m_gpuType = GPUType::Unknown;
    std::wstring m_gpuDescription;       // GPU 描述字符串
    SIZE_T m_gpuDedicatedMemory = 0;     // 专用显存大小（字节）
    SIZE_T m_gpuSharedMemory = 0;        // 共享内存大小（字节）
    std::wstring m_activeAdapterDescription;
    std::wstring m_targetMonitorDeviceName;
    LUID m_activeAdapterLuid = {};
    bool m_hasActiveAdapterLuid = false;
    bool m_isAMDAPU = false;             // 是否是 AMD APU（集成显卡）
    bool m_isIntelIGPU = false;          // 是否是 Intel 核心显卡
    bool m_isLowEndGPU = false;          // 是否是低端 GPU
    int m_intelGeneration = 0;           // Intel 显卡代数（0=未知, 7=第7代, 11=第11代等）

    // GPU 特定优化参数
    int m_optimalBufferCount = 2;        // 最优缓冲区数量
    bool m_useTripleBuffering = false;   // 是否使用三缓冲
    bool m_reduceLatency = false;        // 是否启用低延迟模式
    UINT m_standardSwapChainBufferCount = 2;
    UINT m_mpoSwapChainBufferCount = 2;
    UINT m_standardMaximumFrameLatency = 1;
    UINT m_mpoMaximumFrameLatency = 1;

    int m_targetTextureHoldFrames = 1;      // 每张纹理目标保持的物理刷新次数
    int m_transitionDurationFrames = 1;     // 物理刷新间隔数；1=硬切

    // 参数
    float m_staticAlpha = 0.04f;          // 静态 Cb 模板贴屏强度
    float m_dynamicAlpha = 0.04f;         // 动态 Cr 模板贴屏强度
    int m_refreshRate = 60;              // 当前目标显示器的物理刷新率（Hz）
    int m_currentTextureIndex = 0;
    int m_previousTextureIndex = 0;

    // 状态
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isSuspended{false}; // 暂停渲染标志（用于资源重建期间）
    std::atomic<bool> m_isResizing{false};  // 防止重入的 ResizeBuffers
    std::atomic<bool> m_isRecovering{false}; // 防止重入的设备恢复
    std::atomic<bool> m_ignoreResizeEvents{false}; // 内部调整窗口时忽略同步 WM_SIZE

    std::uint64_t m_totalFrameCount = 0;  // 成功提交计数；结合刷新校正得到纹理阶段
    std::int64_t m_phaseRefreshCorrection = 0; // 物理刷新与 Present 序号的累计偏移
    bool m_displayPhaseAnchorValid = false;
    UINT m_lastPhasePresentId = 0;
    UINT m_lastPhaseRefreshCount = 0;

    // 置顶状态管理
    bool m_needsRestack = false;           // 是否需要重新置顶
    std::chrono::steady_clock::time_point m_lastRestackTime;  // 上次置顶时间
    int m_restackCooldownMs = 500;         // 置顶冷却时间（毫秒）
    int m_consecutivePresentFailures = 0;  // 连续 Present 失败次数
    int m_consecutiveMapFailures = 0;      // 连续常量缓冲区 Map 失败次数
    int m_consecutiveRecoveryFailures = 0; // 连续恢复失败次数
    int m_maxConsecutivePresentFailures = 60;
    int m_maxConsecutiveRecoveryFailures = 3;
    std::atomic<bool> m_fatalShutdownTriggered{false};

    // 设备恢复状态
    int m_deviceRecoveryAttempts = 0;      // 设备恢复尝试次数
    static const int MAX_RECOVERY_ATTEMPTS = 5;  // 最大恢复尝试次数
    std::atomic<int> m_pendingRecoveryRequest{static_cast<int>(RecoveryRequest::None)};
    std::atomic<long> m_pendingRecoveryHr{static_cast<long>(S_OK)};

    // 共享内存遥测
    HANDLE m_telemetryMapping = nullptr;
    DXOverlayTelemetry* m_telemetry = nullptr;

    // Constant buffer data (must match shader cbuffer layout)
    struct alignas(16) ConstantBufferData {
        float staticAlpha;            // 静态 Cb 贴屏强度
        float dynamicAlpha;           // 动态 Cr 贴屏强度
        float textureIndex;           // 当前纹理索引
        float previousTextureIndex;   // 上一张纹理索引
        float transitionBlend;        // 过渡混合系数 [0, 1]
        float channelEncodedTemplates;
        float dynamicUsesSourceAlpha;
        float padding;
    };
    static_assert(sizeof(ConstantBufferData) == 32,
        "ConstantBufferData must match the shader's two 16-byte registers");

    // 错误信息
    std::string m_lastError;
};
