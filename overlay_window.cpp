#include "overlay_internal.h"

bool DXOverlay::CreateOverlayWindow() {
    const wchar_t* className = L"DXOverlayClass";

    // 注册窗口类
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = m_hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = className;
    wc.hbrBackground = nullptr;

    if (!RegisterClassExW(&wc)) {
        DWORD err = ::GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            char msg[256];
            sprintf_s(msg, "Failed to register window class, error: %lu\n", err);
            OutputDebugStringA(msg);
            return false;
        }
    }

    // Pure DirectComposition path:
    // - WS_EX_LAYERED is kept for reliable top-level mouse pass-through
    // - alpha is still provided by the premultiplied composition swap chain
    // - no DWM glass extension, the visual content is the composition surface
    // - WS_EX_TRANSPARENT provides cross-process mouse hit-test pass-through
    // - WM_NCHITTEST returning HTTRANSPARENT remains as a secondary safeguard
    DWORD exStyle =
        WS_EX_TOPMOST |
        WS_EX_TOOLWINDOW |
        WS_EX_LAYERED |
        WS_EX_TRANSPARENT |
        WS_EX_NOACTIVATE |
        WS_EX_NOREDIRECTIONBITMAP;
    DWORD style = WS_POPUP | WS_VISIBLE;

    // 创建全屏窗口（覆盖目标显示器）
    // 使用 m_windowX, m_windowY 支持多显示器
    int windowWidth = m_screenWidth;
    int windowHeight = m_screenHeight;

    char sizeMsg[128];
    sprintf_s(sizeMsg, "Creating fullscreen window: %dx%d at (%d, %d)\n",
              windowWidth, windowHeight, m_windowX, m_windowY);
    LogToFile(sizeMsg);

    m_hwnd = CreateWindowExW(
        exStyle,
        className,
        L"DXOverlay",
        style,
        m_windowX, m_windowY, windowWidth, windowHeight,
        nullptr, nullptr, m_hInstance, this
    );

    if (!m_hwnd) {
        DWORD err = ::GetLastError();
        char msg[256];
        sprintf_s(msg, "Failed to create window, error: %lu\n", err);
        OutputDebugStringA(msg);
        return false;
    }

    // 透明度通过 shader 常量缓冲区传递，不使用 SetLayeredWindowAttributes
    // DirectComposition + 预乘 alpha 提供更稳定的透明度控制
    char alphaMsg[192];
    sprintf_s(
        alphaMsg,
        "Shader channel strengths: static Cb=%.3f, dynamic Cr=%.3f\n",
        m_staticAlpha,
        m_dynamicAlpha);
    LogToFile(alphaMsg);

    ShowWindow(m_hwnd, SW_SHOWNOACTIVATE);
    UpdateWindow(m_hwnd);

    return true;
}

LRESULT CALLBACK DXOverlay::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    DXOverlay* overlay = reinterpret_cast<DXOverlay*>(
        GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTW* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        overlay = create ? static_cast<DXOverlay*>(create->lpCreateParams) : nullptr;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(overlay));
    }

    if (overlay) {
        switch (msg) {
            case WM_NCHITTEST:
                // 返回 HTTRANSPARENT 使所有鼠标事件穿透到下层窗口
                return HTTRANSPARENT;
            case WM_MOUSEACTIVATE:
                // 防止窗口被激活
                return MA_NOACTIVATE;
            case WM_DESTROY:
                overlay->Stop();
                PostQuitMessage(0);
                return 0;
            case WM_NCDESTROY:
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                break;
            case WM_ERASEBKGND:
                return 1; // 防止闪烁
            case WM_SIZE:
                // 处理窗口大小变化
                if (wParam != SIZE_MINIMIZED &&
                    overlay->m_swapChain &&
                    !overlay->m_ignoreResizeEvents.load()) {
                    overlay->OnWindowSizeChanged();
                }
                return 0;
            case WM_DISPLAYCHANGE:
                // 处理显示器配置变化（分辨率、多显示器等）
                overlay->OnDisplayChange();
                return 0;
            case WM_WINDOWPOSCHANGED: {
                const WINDOWPOS* pos = reinterpret_cast<const WINDOWPOS*>(lParam);
                if (pos && !(pos->flags & SWP_NOZORDER)) {
                    overlay->OnZOrderChanged();
                }
                break;
            }
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

void DXOverlay::CheckAndRestoreTopmost() {
    if (!m_hwnd) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!m_needsRestack &&
        std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastRestackTime).count() <
            m_restackCooldownMs) {
        return;
    }

    const LONG_PTR requiredExStyle =
        WS_EX_TOPMOST |
        WS_EX_TOOLWINDOW |
        WS_EX_LAYERED |
        WS_EX_TRANSPARENT |
        WS_EX_NOACTIVATE |
        WS_EX_NOREDIRECTIONBITMAP;
    const LONG_PTR exStyle = GetWindowLongPtrW(m_hwnd, GWL_EXSTYLE);
    const bool topmostWasMissing = (exStyle & WS_EX_TOPMOST) == 0;
    const LONG_PTR repairedExStyle = exStyle | requiredExStyle;
    const bool stylesRepaired = repairedExStyle != exStyle;
    if (stylesRepaired) {
        SetWindowLongPtrW(m_hwnd, GWL_EXSTYLE, repairedExStyle);
        SetWindowPos(
            m_hwnd,
            nullptr,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        LogToFile("Overlay extended styles repaired for click-through stability\n");
    }

    if (!topmostWasMissing) {
        m_needsRestack = false;
        return;
    }

    if (SetWindowPos(
            m_hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOOWNERZORDER)) {
        m_needsRestack = false;
        m_lastRestackTime = now;
        LogToFile("Topmost state restored on demand\n");
    }
}

void DXOverlay::OnZOrderChanged() {
    if (!m_hwnd) {
        return;
    }

    const LONG_PTR exStyle = GetWindowLongPtrW(m_hwnd, GWL_EXSTYLE);
    if ((exStyle & WS_EX_TOPMOST) == 0 ||
        (exStyle & WS_EX_TRANSPARENT) == 0 ||
        (exStyle & WS_EX_LAYERED) == 0) {
        m_needsRestack = true;
    }
}

void DXOverlay::OnWindowSizeChanged() {
    // 窗口大小变化时重新调整 swap chain
    if (!m_swapChain || !m_device || !m_context) {
        return;
    }

    if (m_isRecovering.load()) {
        LogToFile("Window size change ignored during device recovery\n");
        return;
    }

    // 获取新的窗口大小
    RECT clientRect;
    if (!GetClientRect(m_hwnd, &clientRect)) {
        return;
    }

    int newWidth = clientRect.right - clientRect.left;
    int newHeight = clientRect.bottom - clientRect.top;

    if (newWidth <= 0 || newHeight <= 0) {
        return;  // 窗口被最小化或无效大小
    }

    if (!ResizeSwapChainBuffers(newWidth, newHeight, "window size change")) {
        QueueRecovery(
            RecoveryRequest::Presentation,
            DXGI_ERROR_INVALID_CALL,
            "window resize failed");
    }
}

void DXOverlay::OnDisplayChange() {
    // 显示器配置变化时重新检测屏幕尺寸并调整窗口
    LogToFile("Display configuration changed, re-enumerating monitors...\n");

    if (m_isRecovering.load()) {
        LogToFile("Display change ignored during device recovery\n");
        return;
    }

    // 重新枚举显示器
    EnumerateMonitors();

    char monitorMsg[256];
    sprintf_s(monitorMsg, "Display change detected, found %zu monitors\n", m_monitors.size());
    LogToFile(monitorMsg);

    // 优先用稳定的设备名称重定位显示器，避免热插拔后枚举索引变化。
    bool monitorValid = m_targetMonitorIndex == -1;
    if (!monitorValid && !m_targetMonitorDeviceName.empty()) {
        for (size_t i = 0; i < m_monitors.size(); ++i) {
            if (m_monitors[i].name == m_targetMonitorDeviceName) {
                m_targetMonitorIndex = static_cast<int>(i);
                monitorValid = true;
                break;
            }
        }
    }
    if (!monitorValid && m_targetMonitorDeviceName.empty() &&
        m_targetMonitorIndex >= 0 &&
        m_targetMonitorIndex < static_cast<int>(m_monitors.size())) {
        monitorValid = true;
    }

    if (!monitorValid) {
        LogToFile("Target monitor no longer valid, switching to primary\n");
        m_targetMonitorIndex = 0;
    }

    // 重新选择显示器并更新尺寸
    int oldWidth = m_screenWidth;
    int oldHeight = m_screenHeight;
    int oldX = m_windowX;
    int oldY = m_windowY;

    SelectMonitor(m_targetMonitorIndex);

    const bool geometryChanged =
        oldWidth != m_screenWidth || oldHeight != m_screenHeight ||
        oldX != m_windowX || oldY != m_windowY;

    if (geometryChanged) {
        m_ignoreResizeEvents.store(true);
        SetWindowPos(m_hwnd, HWND_TOPMOST, m_windowX, m_windowY,
            m_screenWidth, m_screenHeight, SWP_NOACTIVATE | SWP_NOSENDCHANGING);
        m_ignoreResizeEvents.store(false);

        char msg[256];
        sprintf_s(msg, "Screen updated to %dx%d at (%d, %d)\n",
                  m_screenWidth, m_screenHeight, m_windowX, m_windowY);
        LogToFile(msg);
    }

    // 先检查显卡迁移，不在旧适配器上对 swap chain 执行 ResizeBuffers。
    std::wstring targetAdapterDescription;
    const bool adapterChanged = IsTargetMonitorOnDifferentAdapter(&targetAdapterDescription);
    if (adapterChanged) {
        char msg[512];
        sprintf_s(msg,
            "Display change requires device migration: activeAdapter=%ls, targetAdapter=%ls\n",
            m_activeAdapterDescription.empty() ? L"(unknown)" : m_activeAdapterDescription.c_str(),
            targetAdapterDescription.empty() ? L"(unknown)" : targetAdapterDescription.c_str());
        LogToFile(msg);

        QueueRecovery(
            RecoveryRequest::Device,
            DXGI_ERROR_DEVICE_RESET,
            "target monitor moved to a different adapter");
        return;
    }

    if (geometryChanged &&
        !ResizeSwapChainBuffers(m_screenWidth, m_screenHeight, "display geometry change")) {
        QueueRecovery(
            RecoveryRequest::Presentation,
            DXGI_ERROR_INVALID_CALL,
            "swap-chain resize after display change failed");
        return;
    }

    // 稳定策略始终回到标准呈现路径。
    if (m_mpoFrameLatencyWaitableObject) {
        CloseHandle(m_mpoFrameLatencyWaitableObject);
        m_mpoFrameLatencyWaitableObject = nullptr;
    }
    m_swapChainMPO2.Reset();
    m_swapChainMPO.Reset();
    m_mpoRenderTargetView.Reset();
    m_mpoOutput.Reset();
    m_mpoOverlaySupportFlags = 0;
    m_mpoInitialized = false;
    m_useMPO = false;
    m_mpoCapability = MPOCapability::NotSupported;

    if (!geometryChanged && m_swapChain) {
        if (!SetActiveVisualContent(m_swapChain.Get(), "display change standard path")) {
            return;
        }
    }
}

BOOL CALLBACK DXOverlay::MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    DXOverlay* self = reinterpret_cast<DXOverlay*>(dwData);

    MONITORINFOEXW monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFOEXW);

    if (GetMonitorInfoW(hMonitor, &monitorInfo)) {
        MonitorInfo info;
        info.hMonitor = hMonitor;
        info.rect = monitorInfo.rcMonitor;
        info.width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
        info.height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
        info.refreshRate = 60;
        info.isPrimary = (monitorInfo.dwFlags & MONITORINFOF_PRIMARY) != 0;
        info.name = monitorInfo.szDevice;

        DEVMODEW devMode = {};
        devMode.dmSize = sizeof(devMode);
        if (EnumDisplaySettingsW(monitorInfo.szDevice, ENUM_CURRENT_SETTINGS, &devMode) &&
            devMode.dmDisplayFrequency > 1) {
            info.refreshRate = static_cast<int>(devMode.dmDisplayFrequency);
        }

        self->m_monitors.push_back(info);
    }

    return TRUE;  // 继续枚举
}

void DXOverlay::EnumerateMonitors() {
    m_monitors.clear();
    EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, reinterpret_cast<LPARAM>(this));

    // 确保主显示器在第一位
    for (size_t i = 1; i < m_monitors.size(); ++i) {
        if (m_monitors[i].isPrimary) {
            std::swap(m_monitors[0], m_monitors[i]);
            break;
        }
    }
}

int DXOverlay::GetMonitorCount() {
    if (m_monitors.empty()) {
        EnumerateMonitors();
    }
    return static_cast<int>(m_monitors.size());
}

bool DXOverlay::SelectMonitor(int monitorIndex) {
    if (m_monitors.empty()) {
        // 如果没有枚举过，先枚举
        EnumerateMonitors();
    }

    // 特殊处理：-1 表示覆盖所有显示器（虚拟屏幕）
    if (monitorIndex == -1) {
        m_windowX = GetSystemMetrics(SM_XVIRTUALSCREEN);
        m_windowY = GetSystemMetrics(SM_YVIRTUALSCREEN);
        m_screenWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        m_screenHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        m_currentMonitor = nullptr;
        m_targetMonitorIndex = -1;
        m_targetMonitorDeviceName.clear();

        int minRefreshRate = 1000;
        for (const auto& monitor : m_monitors) {
            if (monitor.refreshRate > 1 && monitor.refreshRate < minRefreshRate) {
                minRefreshRate = monitor.refreshRate;
            }
        }
        m_refreshRate = (minRefreshRate == 1000) ? 60 : minRefreshRate;

        char msg[256];
        sprintf_s(msg, "Selected all monitors (virtual screen): %dx%d at (%d, %d), refresh=%dHz\n",
                  m_screenWidth, m_screenHeight, m_windowX, m_windowY, m_refreshRate);
        LogToFile(msg);
        if (m_monitors.size() > 1) {
            LogToFile(
                "WARNING: monitor_index=-1 uses one virtual-desktop swap chain paced at "
                "the slowest detected refresh rate; mixed-refresh or cross-adapter "
                "monitors cannot be phase-locked independently.\n");
        }
        PublishTelemetryState(m_running.load());
        return true;
    }

    // 验证索引有效性
    if (monitorIndex < 0 || monitorIndex >= static_cast<int>(m_monitors.size())) {
        char msg[128];
        sprintf_s(msg, "Invalid monitor index %d (available: 0-%zu)\n",
                  monitorIndex, m_monitors.size() > 0 ? m_monitors.size() - 1 : 0);
        LogToFile(msg);
        return false;
    }

    const MonitorInfo& monitor = m_monitors[monitorIndex];
    m_windowX = monitor.rect.left;
    m_windowY = monitor.rect.top;
    m_screenWidth = monitor.width;
    m_screenHeight = monitor.height;
    m_currentMonitor = monitor.hMonitor;
    m_targetMonitorIndex = monitorIndex;
    m_targetMonitorDeviceName = monitor.name;
    m_refreshRate = (monitor.refreshRate > 1) ? monitor.refreshRate : 60;

    char msg[256];
    sprintf_s(msg, "Selected monitor %d: %dx%d at (%d, %d), refresh=%dHz %s\n",
              monitorIndex, m_screenWidth, m_screenHeight, m_windowX, m_windowY,
              m_refreshRate, monitor.isPrimary ? "[PRIMARY]" : "");
    LogToFile(msg);
    PublishTelemetryState(m_running.load());

    return true;
}
