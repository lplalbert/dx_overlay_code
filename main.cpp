#include "overlay.h"
#include <windows.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <shellapi.h>
#include <cstdarg>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>
#include <shellscalingapi.h>  // For SetProcessDpiAwareness

// 调试模式开关（发布版设为 false）
#define DEBUG_MODE false

#if DEBUG_MODE
    // 调试模式：启用 printf
#else
    // 发布模式：禁用 printf/wprintf
    #define printf(...) ((void)0)
    #define wprintf(...) ((void)0)
#endif

// 显示消息框（用于关键信息，即使无控制台也能看到）
void ShowMessage(const char* title, const char* msg) {
    MessageBoxA(nullptr, msg, title, MB_OK | MB_ICONINFORMATION);
}

void ShowError(const char* title, const char* msg) {
    MessageBoxA(nullptr, msg, title, MB_OK | MB_ICONERROR);
}

namespace fs = std::filesystem;

namespace {
constexpr wchar_t kSingleInstanceMutexName[] = L"Local\\DXOverlay.SingleInstance.5D4F0C8B-4B84-4F7A-A7F6-7E9A0B5A1F31";
constexpr int kWatermarkPayloadHexDigits = 5;
constexpr int kMaxEncodableWatermarkId = (1 << (4 * kWatermarkPayloadHexDigits)) - 1;  // RS(15,5) over GF(16)

struct ScopedHandle {
    HANDLE handle = nullptr;

    ScopedHandle() = default;
    explicit ScopedHandle(HANDLE value) : handle(value) {}

    ~ScopedHandle() {
        if (handle) {
            CloseHandle(handle);
            handle = nullptr;
        }
    }

    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;
};

enum class SingleInstanceState {
    Acquired,
    AlreadyRunning,
    Failed
};

SingleInstanceState AcquireSingleInstanceGuard(ScopedHandle& guard) {
    HANDLE mutexHandle = CreateMutexW(nullptr, FALSE, kSingleInstanceMutexName);
    if (!mutexHandle) {
        return SingleInstanceState::Failed;
    }

    guard.handle = mutexHandle;
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        return SingleInstanceState::AlreadyRunning;
    }

    return SingleInstanceState::Acquired;
}

void StartupLogf(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsprintf_s(buffer, format, args);
    va_end(args);
    WriteStartupLog(buffer);
}

struct DisplayGeometry {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int resolvedMonitorIndex = 0;
    bool spansAllMonitors = false;
    bool isPrimary = false;
    std::wstring name;
};

BOOL CALLBACK DisplayMonitorEnumProc(HMONITOR hMonitor, HDC, LPRECT, LPARAM dwData) {
    auto* displays = reinterpret_cast<std::vector<DisplayGeometry>*>(dwData);
    if (!displays) {
        return FALSE;
    }

    MONITORINFOEXW monitorInfo = {};
    monitorInfo.cbSize = sizeof(monitorInfo);
    if (!GetMonitorInfoW(hMonitor, &monitorInfo)) {
        return TRUE;
    }

    DisplayGeometry display;
    display.x = monitorInfo.rcMonitor.left;
    display.y = monitorInfo.rcMonitor.top;
    display.width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
    display.height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
    display.isPrimary = (monitorInfo.dwFlags & MONITORINFOF_PRIMARY) != 0;
    display.name = monitorInfo.szDevice;

    displays->push_back(display);
    return TRUE;
}

std::vector<DisplayGeometry> EnumerateDisplays() {
    std::vector<DisplayGeometry> displays;
    EnumDisplayMonitors(nullptr, nullptr, DisplayMonitorEnumProc, reinterpret_cast<LPARAM>(&displays));

    for (size_t i = 1; i < displays.size(); ++i) {
        if (displays[i].isPrimary) {
            std::swap(displays[0], displays[i]);
            break;
        }
    }

    for (size_t i = 0; i < displays.size(); ++i) {
        displays[i].resolvedMonitorIndex = static_cast<int>(i);
    }

    return displays;
}

DisplayGeometry ResolveTargetDisplayGeometry(int requestedMonitorIndex) {
    DisplayGeometry geometry;
    std::vector<DisplayGeometry> displays = EnumerateDisplays();

    if (requestedMonitorIndex == -1) {
        geometry.x = GetSystemMetrics(SM_XVIRTUALSCREEN);
        geometry.y = GetSystemMetrics(SM_YVIRTUALSCREEN);
        geometry.width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        geometry.height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        geometry.resolvedMonitorIndex = -1;
        geometry.spansAllMonitors = true;
        geometry.name = L"VirtualScreen";

        return geometry;
    }

    if (displays.empty()) {
        geometry.width = GetSystemMetrics(SM_CXSCREEN);
        geometry.height = GetSystemMetrics(SM_CYSCREEN);
        geometry.isPrimary = true;
        geometry.name = L"PrimaryScreen";
        geometry.resolvedMonitorIndex = 0;
        return geometry;
    }

    int resolvedIndex = requestedMonitorIndex;
    if (resolvedIndex < 0 || resolvedIndex >= static_cast<int>(displays.size())) {
        resolvedIndex = 0;
    }

    geometry = displays[resolvedIndex];
    geometry.resolvedMonitorIndex = resolvedIndex;
    return geometry;
}
}

// 配置结构
struct Config {
    float alpha = 0.05f;              // 旧配置兼容：未单独设置时同时作用于两路
    float static_alpha = 0.05f;       // 静态 Cb 模板贴屏强度
    float dynamic_alpha = 0.05f;      // 动态 Cr 模板贴屏强度
    int watermark_id = -1;          // 水印编号，-1 表示不自动生成
    int screen_width = 1920;        // 屏幕宽度
    int screen_height = 1080;       // 屏幕高度
    int block_rows = 4;             // 水印块行数
    int block_cols = 6;             // 水印块列数
    int dynamic_ratio_cr = 10;      // 动态 Cr 模板生成幅度（旧名 ratio_u）
    int static_ratio_cb = 8;        // 静态 Cb 模板生成幅度（旧名 ratio_v）
    int type_val = 0;               // 水印图案类型 (0 或 1)
    int train_codeword = -1;        // 训练模式码字（-1 表示关闭）
    int switch_hold_frames = 1;     // 每张纹理目标保持多少个物理刷新周期
    int transition_frames = 1;      // 纹理切换跨多少个物理刷新间隔完成
    int monitor_index = 0;          // 目标显示器索引 (0=主显示器, -1=所有显示器)
    std::string pattern = "rect";          // 静态 Cb 子块图案（动态 Cr 固定矩形）
    std::string generator_mode = "auto";   // auto / rs_script / packed_exe / slim_script
    std::string python_path = "python";  // Python 解释器路径
};

enum class GeneratorKind {
    PackedExe,
    RSPython,
    SlimPython
};

struct GeneratorSelection {
    GeneratorKind kind = GeneratorKind::PackedExe;
    fs::path path;
    bool usesPython = false;
};

void AlignConfigToTargetDisplay(Config& config) {
    const int requestedMonitorIndex = config.monitor_index;
    DisplayGeometry geometry = ResolveTargetDisplayGeometry(requestedMonitorIndex);

    if (geometry.width <= 0 || geometry.height <= 0) {
        StartupLogf("Display resolution lookup failed, keeping configured watermark size %dx%d\n",
            config.screen_width, config.screen_height);
        return;
    }

    if (requestedMonitorIndex != geometry.resolvedMonitorIndex) {
        StartupLogf("Requested monitor %d resolved to monitor %d for runtime\n",
            requestedMonitorIndex, geometry.resolvedMonitorIndex);
        config.monitor_index = geometry.resolvedMonitorIndex;
    }

    if (config.screen_width != geometry.width || config.screen_height != geometry.height) {
        StartupLogf("Watermark generation size aligned to target display: %dx%d -> %dx%d (%ls)\n",
            config.screen_width,
            config.screen_height,
            geometry.width,
            geometry.height,
            geometry.name.c_str());
    } else {
        StartupLogf("Watermark generation size already matches target display: %dx%d (%ls)\n",
            geometry.width,
            geometry.height,
            geometry.name.c_str());
    }

    config.screen_width = geometry.width;
    config.screen_height = geometry.height;
}

namespace {
std::string TrimConfigText(const std::string& text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool TryParseConfigInt(const std::string& text, int& result) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (errno == ERANGE || end == text.c_str() || !end || *end != '\0' ||
        value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        return false;
    }
    result = static_cast<int>(value);
    return true;
}

bool TryParseConfigFloat(const std::string& text, float& result) {
    errno = 0;
    char* end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (errno == ERANGE || end == text.c_str() || !end || *end != '\0' || !std::isfinite(value)) {
        return false;
    }
    result = value;
    return true;
}
}

// 读取配置文件：单个非法字段只回退该字段的默认值，不让进程因配置损坏崩溃。
Config LoadConfig(const fs::path& configPath) {
    Config config;
    bool legacyAlphaSeen = false;
    bool staticAlphaSeen = false;
    bool dynamicAlphaSeen = false;

    std::ifstream file(configPath);
    if (!file.is_open()) {
        StartupLogf("Config file not found; using built-in defaults\n");
        return config;
    }

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }

        line = TrimConfigText(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            StartupLogf("WARNING: Ignoring malformed config line %d (missing '=')\n", lineNumber);
            continue;
        }

        std::string key = TrimConfigText(line.substr(0, pos));
        std::string value = TrimConfigText(line.substr(pos + 1));
        if (key.empty() || value.empty()) {
            StartupLogf("WARNING: Ignoring empty config field at line %d\n", lineNumber);
            continue;
        }

        auto setInt = [&](int& target, int minimum, int maximum) {
            int parsed = 0;
            if (!TryParseConfigInt(value, parsed)) {
                StartupLogf("WARNING: Invalid integer for '%s' at line %d; keeping %d\n",
                    key.c_str(), lineNumber, target);
                return;
            }
            const int clamped = std::clamp(parsed, minimum, maximum);
            if (clamped != parsed) {
                StartupLogf("WARNING: Clamped '%s' from %d to %d at line %d\n",
                    key.c_str(), parsed, clamped, lineNumber);
            }
            target = clamped;
        };

        auto setFloat = [&](float& target, float minimum, float maximum) {
            float parsed = 0.0f;
            if (!TryParseConfigFloat(value, parsed)) {
                StartupLogf("WARNING: Invalid number for '%s' at line %d; keeping %.4f\n",
                    key.c_str(), lineNumber, target);
                return false;
            }
            const float clamped = std::clamp(parsed, minimum, maximum);
            if (clamped != parsed) {
                StartupLogf("WARNING: Clamped '%s' from %.4f to %.4f at line %d\n",
                    key.c_str(), parsed, clamped, lineNumber);
            }
            target = clamped;
            return true;
        };

        if (key == "alpha") {
            legacyAlphaSeen = setFloat(config.alpha, 0.0f, 1.0f) || legacyAlphaSeen;
        } else if (key == "static_alpha") {
            staticAlphaSeen = setFloat(config.static_alpha, 0.0f, 1.0f) || staticAlphaSeen;
        } else if (key == "dynamic_alpha") {
            dynamicAlphaSeen = setFloat(config.dynamic_alpha, 0.0f, 1.0f) || dynamicAlphaSeen;
        } else if (key == "watermark_id") {
            setInt(config.watermark_id, -1, kMaxEncodableWatermarkId);
        } else if (key == "block_rows") {
            setInt(config.block_rows, 1, 64);
        } else if (key == "block_cols") {
            setInt(config.block_cols, 1, 64);
        } else if (key == "dynamic_ratio_cr" || key == "ratio_u") {
            setInt(config.dynamic_ratio_cr, 0, 255);
        } else if (key == "static_ratio_cb" || key == "ratio_v") {
            setInt(config.static_ratio_cb, 0, 255);
        } else if (key == "type_val") {
            setInt(config.type_val, 0, 1);
        } else if (key == "train_codeword") {
            setInt(config.train_codeword, -1, 15);
        } else if (key == "switch_hold_frames") {
            setInt(config.switch_hold_frames, 1, 16);
        } else if (key == "transition_frames") {
            setInt(config.transition_frames, 1, 16);
        } else if (key == "monitor_index") {
            setInt(config.monitor_index, -1, 63);
        } else if (key == "pattern") {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (value == "gaussian" || value == "gaussian_v2" ||
                value == "soft_rect" || value == "rect") {
                config.pattern = value;
            } else {
                StartupLogf("WARNING: Unsupported pattern '%s' at line %d; keeping '%s'\n",
                    value.c_str(), lineNumber, config.pattern.c_str());
            }
        } else if (key == "generator_mode") {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (value == "auto" || value == "rs_script" || value == "packed_exe" ||
                value == "slim_script") {
                config.generator_mode = value;
            } else {
                StartupLogf("WARNING: Unsupported generator_mode '%s' at line %d; keeping '%s'\n",
                    value.c_str(), lineNumber, config.generator_mode.c_str());
            }
        } else if (key == "python_path") {
            config.python_path = value;
        } else {
            StartupLogf("WARNING: Ignoring unknown config key '%s' at line %d\n",
                key.c_str(), lineNumber);
        }
    }

    // 旧 alpha 是两路强度的快捷方式。通道专用配置优先且不受行顺序影响。
    if (legacyAlphaSeen && !staticAlphaSeen) {
        config.static_alpha = config.alpha;
    }
    if (legacyAlphaSeen && !dynamicAlphaSeen) {
        config.dynamic_alpha = config.alpha;
    }

    StartupLogf(
        "Config parsed safely: staticCbAlpha=%.4f, dynamicCrAlpha=%.4f, watermark_id=%d, monitor=%d, blocks=%dx%d, hold=%d\n",
        config.static_alpha,
        config.dynamic_alpha,
        config.watermark_id,
        config.monitor_index,
        config.block_rows,
        config.block_cols,
        config.switch_hold_frames);
    return config;
}

bool TryFindGeneratorScript(const std::vector<fs::path>& roots, const char* scriptName, fs::path& outPath) {
    for (const auto& root : roots) {
        if (root.empty()) {
            continue;
        }
        fs::path candidate = root / scriptName;
        if (fs::exists(candidate)) {
            outPath = candidate;
            return true;
        }
    }
    return false;
}

bool SelectGeneratorExecutable(const Config& config, const fs::path& exeDir, GeneratorSelection& selection) {
    const fs::path generatorExe = exeDir / "wm_generator.exe";
    const std::vector<fs::path> searchRoots = {
        exeDir,
        exeDir.parent_path(),
        exeDir.parent_path().parent_path()
    };

    fs::path rsScript;
    fs::path slimScript;
    const bool hasRsScript = TryFindGeneratorScript(searchRoots, "rs_gen_Syn_template_nums_dual.py", rsScript);
    const bool hasSlimScript = TryFindGeneratorScript(searchRoots, "wm_generator_slim.py", slimScript);
    const bool hasPackedExe = fs::exists(generatorExe);

    if (config.generator_mode == "rs_script") {
        if (hasRsScript) {
            selection = {GeneratorKind::RSPython, rsScript, true};
            return true;
        }
        return false;
    }

    if (config.generator_mode == "packed_exe") {
        if (hasPackedExe) {
            selection = {GeneratorKind::PackedExe, generatorExe, false};
            return true;
        }
        return false;
    }

    if (config.generator_mode == "slim_script") {
        if (hasSlimScript) {
            selection = {GeneratorKind::SlimPython, slimScript, true};
            return true;
        }
        return false;
    }

    if (hasPackedExe) {
        selection = {GeneratorKind::PackedExe, generatorExe, false};
        return true;
    }
    if (hasRsScript) {
        selection = {GeneratorKind::RSPython, rsScript, true};
        return true;
    }
    if (hasSlimScript) {
        selection = {GeneratorKind::SlimPython, slimScript, true};
        return true;
    }

    return false;
}

// 调用水印生成器（优先使用打包的 exe，否则使用 Python 脚本）
bool GenerateWatermarkImages(const Config& config, const fs::path& exeDir) {
    if (config.watermark_id < 0) {
        return false;  // 未指定水印编号，不生成
    }

    if (config.watermark_id > kMaxEncodableWatermarkId) {
        char errMsg[768];
        sprintf_s(
            errMsg,
            "watermark_id=%d is too large for the current RS(15,5) generator.\n\n"
            "The current payload has %d hexadecimal symbols, so the supported range is:\n"
            "  0 to %d (0x%05X)\n\n"
            "Use a smaller numeric ID, or redesign both the encoder and decoder for a larger payload.",
            config.watermark_id,
            kWatermarkPayloadHexDigits,
            kMaxEncodableWatermarkId,
            kMaxEncodableWatermarkId);
        ShowError("Watermark ID Out of Range", errMsg);
        StartupLogf(
            "ERROR: watermark_id=%d exceeds supported RS(15,5) payload range 0..%d\n",
            config.watermark_id,
            kMaxEncodableWatermarkId);
        return false;
    }

    // 构建输出目录
    fs::path outputDir = exeDir / "wm_imgs";
    fs::create_directories(outputDir);

    GeneratorSelection generator;
    if (!SelectGeneratorExecutable(config, exeDir, generator)) {
        printf("ERROR: no compatible watermark generator found for generator_mode=%s\n",
            config.generator_mode.c_str());
        return false;
    }

    // 构建命令行
    char cmd[4096];
    if (!generator.usesPython) {
        const bool useTrainMode = config.train_codeword >= 0;
        const std::string trainArg = useTrainMode
            ? " --train " + std::to_string(config.train_codeword)
            : "";
        sprintf_s(cmd, "\"%s\" --nums %d --block_rows %d --block_cols %d "
                  "--screen_width %d --screen_height %d "
                  "--pattern %s "
                  "--dynamic_ratio_cr %d --static_ratio_cb %d --type_val %d --save_dir \"%s\"%s",
                  generator.path.string().c_str(),
                  config.watermark_id,
                  config.block_rows,
                  config.block_cols,
                  config.screen_width,
                  config.screen_height,
                  config.pattern.c_str(),
                  config.dynamic_ratio_cr,
                  config.static_ratio_cb,
                  config.type_val,
                  outputDir.string().c_str(),
                  trainArg.c_str());
        printf("Using packed generator: wm_generator.exe\n");
        StartupLogf("Using packed generator: %s\n", generator.path.string().c_str());
    } else {
        switch (generator.kind) {
            case GeneratorKind::RSPython: {
                const bool useTrainMode = config.train_codeword >= 0;
                const std::string trainArg = useTrainMode
                    ? " --train " + std::to_string(config.train_codeword)
                    : "";
                sprintf_s(cmd, "\"%s\" \"%s\" --nums %d --block_rows %d --block_cols %d "
                          "--screen_width %d --screen_height %d "
                          "--pattern %s "
                          "--dynamic_ratio_cr %d --static_ratio_cb %d --type_val %d --save_dir \"%s\"%s",
                          config.python_path.c_str(),
                          generator.path.string().c_str(),
                          config.watermark_id,
                          config.block_rows,
                          config.block_cols,
                          config.screen_width,
                          config.screen_height,
                          config.pattern.c_str(),
                          config.dynamic_ratio_cr,
                          config.static_ratio_cb,
                          config.type_val,
                          outputDir.string().c_str(),
                          trainArg.c_str());
                break;
            }
            case GeneratorKind::SlimPython:
                sprintf_s(cmd, "\"%s\" \"%s\" --nums %d --block_rows %d --block_cols %d "
                          "--screen_width %d --screen_height %d "
                          "--dynamic_ratio_cr %d --static_ratio_cb %d --type_val %d --pattern %s --save_dir \"%s\"",
                          config.python_path.c_str(),
                          generator.path.string().c_str(),
                          config.watermark_id,
                          config.block_rows,
                          config.block_cols,
                          config.screen_width,
                          config.screen_height,
                          config.dynamic_ratio_cr,
                          config.static_ratio_cb,
                          config.type_val,
                          config.pattern.c_str(),
                          outputDir.string().c_str());
                break;
            case GeneratorKind::PackedExe:
                break;
        }

        printf("Using Python script: %s\n", generator.path.string().c_str());
        StartupLogf("Using Python generator script: %s\n", generator.path.string().c_str());
    }

    printf("Command: %s\n", cmd);
    StartupLogf("Generator command: %s\n", cmd);

    // 使用 CreateProcess 来执行生成器（比 system() 更可靠）
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // 创建子进程
    BOOL success = CreateProcessA(
        nullptr,           // 应用程序名
        cmd,              // 命令行
        nullptr,          // 进程安全属性
        nullptr,          // 线程安全属性
        FALSE,            // 继承句柄
        CREATE_NO_WINDOW, // 不创建窗口
        nullptr,          // 环境变量
        outputDir.string().c_str(),  // 工作目录设为输出目录
        &si,              // STARTUPINFO
        &pi               // PROCESS_INFORMATION
    );

    int result = -1;
    if (success) {
        // 等待进程完成（最多30秒）
        DWORD waitResult = WaitForSingleObject(pi.hProcess, 30000);
        if (waitResult == WAIT_OBJECT_0) {
            DWORD exitCode;
            GetExitCodeProcess(pi.hProcess, &exitCode);
            result = (int)exitCode;
        } else {
            // 超时，终止进程
            TerminateProcess(pi.hProcess, 1);
            result = -2;  // 超时
        }
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        DWORD error = GetLastError();
        char errMsg[512];
        sprintf_s(errMsg, "Failed to start watermark generator\nError code: %lu\n\nCommand:\n%s", error, cmd);
        ShowError("Process Creation Failed", errMsg);
        return false;
    }

    if (result != 0) {
        char errMsg[1024];
        if (!generator.usesPython) {
            sprintf_s(errMsg, "wm_generator.exe failed with code %d\n\nCommand:\n%s\n\nCheck wm_generator_error.log for details", result, cmd);
            ShowError("Watermark Generation Failed", errMsg);
        } else {
            const char* dependencyHint =
                (generator.kind == GeneratorKind::SlimPython)
                    ? "numpy, opencv-python, galois"
                    : "numpy, opencv-python, reedsolo";
            sprintf_s(errMsg, "Python script failed with code %d\n\nPlease ensure Python is installed with:\n%s",
                result, dependencyHint);
            ShowError("Watermark Generation Failed", errMsg);
        }
        return false;
    }

    // 检查生成的文件
    char filename1[256], filename2[256];
    sprintf_s(filename1, "wm_template_%d.png", config.watermark_id);
    sprintf_s(filename2, "wm_template_%d_inverse.png", config.watermark_id);

    fs::path img1 = outputDir / filename1;
    fs::path img2 = outputDir / filename2;

    if (fs::exists(img1) && fs::exists(img2)) {
        printf("Watermark images generated successfully:\n");
        printf("  %s\n", img1.string().c_str());
        printf("  %s\n", img2.string().c_str());
        return true;
    } else {
        printf("ERROR: Generated watermark images not found\n");
        return false;
    }
}

// Windows GUI 应用程序入口点
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    (void)hInstance;
    (void)hPrevInstance;
    (void)lpCmdLine;
    (void)nCmdShow;

    // 获取可执行文件目录
    wchar_t exePathBuf[MAX_PATH];
    GetModuleFileNameW(nullptr, exePathBuf, MAX_PATH);
    fs::path exeDir = fs::path(exePathBuf).parent_path();

    ScopedHandle singleInstanceGuard;
    const SingleInstanceState instanceState = AcquireSingleInstanceGuard(singleInstanceGuard);
    if (instanceState == SingleInstanceState::AlreadyRunning) {
        OutputDebugStringA("DX Overlay single-instance guard: another instance is already running\n");
        return 0;
    }
    if (instanceState == SingleInstanceState::Failed) {
        ShowError("DX Overlay", "Failed to initialize the single-instance guard.\n\nThe application will exit to avoid launching duplicate overlays.");
        return -1;
    }

    // 解析命令行参数
    int argc = 0;
    LPWSTR cmdLineW = GetCommandLineW();
    LPWSTR* argv = CommandLineToArgvW(cmdLineW, &argc);
    bool enableVerboseLogging = false;
    for (int i = 1; argv && i < argc; ++i) {
        if (_wcsicmp(argv[i], L"log") == 0 || _wcsicmp(argv[i], L"-log") == 0 || _wcsicmp(argv[i], L"--log") == 0) {
            enableVerboseLogging = true;
            break;
        }
    }

    InitializeLogging(exeDir.wstring(), enableVerboseLogging);
    StartupLogf("Executable directory: %s\n", exeDir.string().c_str());
    StartupLogf("Command line logging flag: %s\n", enableVerboseLogging ? "enabled" : "disabled");

    // 使用保守的优先级策略，避免抢占 DWM、驱动和系统线程导致不稳定
    if (SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)) {
        printf("Process priority set to ABOVE_NORMAL\n");
        StartupLogf("Process priority set to ABOVE_NORMAL\n");
    } else {
        printf("Warning: Failed to raise process priority\n");
        StartupLogf("Warning: Failed to raise process priority\n");
    }

    if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL)) {
        printf("Main thread priority set to ABOVE_NORMAL\n");
        StartupLogf("Main thread priority set to ABOVE_NORMAL\n");
    } else {
        printf("Warning: Failed to raise main thread priority\n");
        StartupLogf("Warning: Failed to raise main thread priority\n");
    }

    // 防止系统休眠和显示器关闭（电池供电时尤其重要）
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    printf("System sleep prevention enabled\n");
    StartupLogf("System sleep prevention enabled\n");

    // 设置 DPI 感知，获取真实屏幕分辨率
    SetProcessDPIAware();  // Windows Vista+
    printf("DPI awareness set\n");
    StartupLogf("DPI awareness set\n");

    // 初始化 COM（用于 WIC 图像加载）
    HRESULT comResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    printf("COM initialized: 0x%08X\n", comResult);
    StartupLogf("COM initialized: 0x%08X\n", static_cast<unsigned int>(comResult));

    // 默认图像路径（相对于可执行文件）
    std::vector<std::wstring> imagePaths;

    if (argv) {
        LocalFree(argv);
    }

    // 先加载配置文件（需要知道 watermark_id 来决定是否生成水印）
    fs::path configPath = exeDir / "config.txt";
    StartupLogf("Loading config from: %s\n", configPath.string().c_str());
    Config config = LoadConfig(configPath);
    StartupLogf(
        "Config loaded: staticCbAlpha=%.4f, dynamicCrAlpha=%.4f, watermark_id=%d, monitor_index=%d, hold=%d, transition=%d, pattern=%s\n",
        config.static_alpha,
        config.dynamic_alpha,
        config.watermark_id,
        config.monitor_index,
        config.switch_hold_frames,
        config.transition_frames,
        config.pattern.c_str());
    AlignConfigToTargetDisplay(config);
    StartupLogf("Runtime target display config: monitor_index=%d, screen=%dx%d\n",
        config.monitor_index, config.screen_width, config.screen_height);

    // 如果指定了 watermark_id，先生成水印图像
    if (config.watermark_id >= 0) {
        printf("Watermark ID specified: %d, generating watermark images...\n", config.watermark_id);
        StartupLogf("Watermark generation requested for id=%d\n", config.watermark_id);
        if (!GenerateWatermarkImages(config, exeDir)) {
            printf("WARNING: Failed to generate watermark images, will try to use existing ones\n");
            StartupLogf("WARNING: Failed to generate watermark images, falling back to existing files\n");
        } else {
            StartupLogf("Watermark images generated successfully\n");
        }
    } else {
        // watermark_id 为 -1，使用已有图像
        printf("No watermark_id specified, using existing images in wm_imgs folder\n");
        StartupLogf("No watermark_id specified, using existing images in wm_imgs folder\n");
    }

    {
        // 查找 wm_imgs 目录（尝试多个可能的位置）
        fs::path imgDir;
        std::vector<fs::path> possibleDirs = {
            exeDir / "wm_imgs",
            exeDir.parent_path() / "wm_imgs",
            exeDir.parent_path().parent_path() / "wm_imgs"
        };

        for (const auto& dir : possibleDirs) {
            if (fs::exists(dir) && fs::is_directory(dir)) {
                imgDir = dir;
                break;
            }
        }

        // 如果指定了 watermark_id，优先使用生成的水印图像
        if (config.watermark_id >= 0 && !imgDir.empty()) {
            char filename1[256], filename2[256];
            sprintf_s(filename1, "wm_template_%d.png", config.watermark_id);
            sprintf_s(filename2, "wm_template_%d_inverse.png", config.watermark_id);

            fs::path img1 = imgDir / filename1;
            fs::path img2 = imgDir / filename2;

            if (fs::exists(img1) && fs::exists(img2)) {
                // 使用指定编号的水印图像
                try {
                    imagePaths.push_back(fs::canonical(img1).wstring());
                    imagePaths.push_back(fs::canonical(img2).wstring());
                } catch (const std::exception&) {
                    imagePaths.push_back(fs::absolute(img1).wstring());
                    imagePaths.push_back(fs::absolute(img2).wstring());
                }
                printf("Using watermark images for ID %d\n", config.watermark_id);
            }
        }

        // 如果没有找到指定编号，则回退到目录中的第一组正式正/反模板。
        if (imagePaths.empty() && !imgDir.empty() && fs::exists(imgDir)) {
            std::vector<std::pair<fs::path, fs::path>> imagePairs;
            for (const auto& entry : fs::directory_iterator(imgDir)) {
                if (entry.is_regular_file()) {
                    std::wstring ext = entry.path().extension().wstring();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
                    const std::wstring stem = entry.path().stem().wstring();
                    const std::wstring prefix = L"wm_template_";
                    const std::wstring inverseSuffix = L"_inverse";
                    if (ext == L".png" && stem.rfind(prefix, 0) == 0 &&
                        stem.size() > prefix.size() &&
                        (stem.size() < inverseSuffix.size() ||
                         stem.compare(stem.size() - inverseSuffix.size(),
                                      inverseSuffix.size(), inverseSuffix) != 0)) {
                        const fs::path inverse = entry.path().parent_path() /
                            (stem + inverseSuffix + L".png");
                        if (fs::exists(inverse) && fs::is_regular_file(inverse)) {
                            imagePairs.emplace_back(entry.path(), inverse);
                        }
                    }
                }
            }

            std::sort(imagePairs.begin(), imagePairs.end(), [](const auto& a, const auto& b) {
                return a.first.filename().wstring() < b.first.filename().wstring();
            });

            if (!imagePairs.empty()) {
                for (const fs::path& imagePath : {imagePairs.front().first, imagePairs.front().second}) {
                    try {
                        fs::path absPath = fs::canonical(imagePath);
                        imagePaths.push_back(absPath.wstring());
                    } catch (const std::exception&) {
                        imagePaths.push_back(fs::absolute(imagePath).wstring());
                    }
                }
            }
        }
    }

    if (imagePaths.empty()) {
        printf("ERROR: No image files found!\n");
        StartupLogf("ERROR: No image files found during startup\n");

        // 构建详细的错误信息
        wchar_t errDetail[1024];
        swprintf_s(errDetail,
            L"No watermark images found!\n\n"
            L"Current config:\n"
            L"  watermark_id = %d\n\n"
            L"To fix this, edit config.txt:\n"
            L"  1. Set watermark_id to your ID (e.g., watermark_id=123456)\n"
            L"  2. Run dx_overlay.exe again\n\n"
            L"Or manually place images in wm_imgs folder.",
            config.watermark_id);
        MessageBoxW(nullptr, errDetail, L"Error - No Images", MB_OK | MB_ICONERROR);
        CompleteStartupLogging();
        CoUninitialize();
        return -1;
    }

    // 如果只有一张图片，复制一份
    if (imagePaths.size() == 1) {
        imagePaths.push_back(imagePaths[0]);
    }

    printf("Found %zu images:\n", imagePaths.size());
    StartupLogf("Selected %zu image(s) for overlay\n", imagePaths.size());
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        wprintf(L"  [%zu] %s\n", i, imagePaths[i].c_str());
    }
    // 验证图像文件是否存在
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        if (!fs::exists(imagePaths[i])) {
            wprintf(L"ERROR: Image file does not exist: %s\n", imagePaths[i].c_str());
            StartupLogf("ERROR: Image file does not exist at startup index=%zu\n", i);
            wchar_t errorMsg[512];
            swprintf_s(errorMsg, L"Image file does not exist:\n%s", imagePaths[i].c_str());
            MessageBoxW(nullptr, errorMsg, L"Error", MB_OK | MB_ICONERROR);
            CompleteStartupLogging();
            CoUninitialize();
            return -1;
        }
    }

    printf("All images verified.\n");

    // 输出图像路径信息（用于调试）
    wchar_t debugMsg[4096];
    swprintf_s(debugMsg, L"Loading %zu image(s):\n", imagePaths.size());
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        wchar_t pathMsg[2048];
        bool exists = fs::exists(imagePaths[i]);
        // 使用 %ls 来正确显示宽字符串
        swprintf_s(pathMsg, L"  %zu: %ls [%s]\n", i, imagePaths[i].c_str(), exists ? L"EXISTS" : L"NOT FOUND");
        wcscat_s(debugMsg, pathMsg);
    }
    OutputDebugStringW(debugMsg);

    // 创建并初始化 Overlay
    printf("Creating DXOverlay...\n");
    StartupLogf("Creating DXOverlay instance\n");
    DXOverlay overlay;

    // 设置目标显示器（在 Initialize 之前）
    int monitorCount = overlay.GetMonitorCount();
    printf("Found %d monitor(s), target monitor: %d\n", monitorCount, config.monitor_index);
    StartupLogf("Detected %d monitor(s), target monitor=%d\n", monitorCount, config.monitor_index);
    if (config.monitor_index == -1 && monitorCount > 1) {
        printf("WARNING: virtual-desktop mode uses one swap chain at the slowest "
               "detected refresh rate; mixed-refresh monitors are not independently synchronized.\n");
    }
    overlay.SetTargetMonitor(config.monitor_index);
    overlay.SetSwitchingBehavior(
        config.switch_hold_frames,
        config.transition_frames);
    StartupLogf(
        "Applied switching behavior: hold=%d display refresh(es), transition=%d refresh interval(s)\n",
        config.switch_hold_frames,
        config.transition_frames);

    // 使用配置文件中的参数初始化
    printf("Initializing overlay with static Cb alpha=%.4f, dynamic Cr alpha=%.4f...\n",
        config.static_alpha,
        config.dynamic_alpha);
    StartupLogf(
        "Initializing overlay strengths: static Cb=%.4f, dynamic Cr=%.4f\n",
        config.static_alpha,
        config.dynamic_alpha);
    if (!overlay.Initialize(imagePaths, config.static_alpha, config.dynamic_alpha)) {
        // 获取错误信息
        std::string error = overlay.GetLastError();
        printf("ERROR: Initialize failed: %s\n", error.c_str());
        StartupLogf("ERROR: Overlay initialization failed: %s\n", error.c_str());

        // 将错误信息转换为宽字符串
        std::wstring errorW;
        if (!error.empty()) {
            int size = MultiByteToWideChar(CP_UTF8, 0, error.c_str(), -1, nullptr, 0);
            if (size > 0) {
                errorW.resize(size - 1);
                MultiByteToWideChar(CP_UTF8, 0, error.c_str(), -1, &errorW[0], size);
            }
        }
        if (errorW.empty()) {
            errorW = L"Unknown error";
        }

        // 构建错误消息
        std::wstring errorMsg = L"Failed to initialize DirectX Overlay!\n\nError: " + errorW + L"\n\nCheck Debug Output for more details.\n\nImages tried:\n";
        for (size_t i = 0; i < imagePaths.size() && i < 3; ++i) {
            errorMsg += imagePaths[i] + L"\n";
        }

        MessageBoxW(nullptr, errorMsg.c_str(), L"Error", MB_OK | MB_ICONERROR);
        printf("Press Enter to exit...\n");
        getchar();
        CompleteStartupLogging();
        CoUninitialize();
        return -1;
    }

    // 运行主循环
    printf("Running overlay... (Press ESC to exit)\n");
    StartupLogf("Overlay initialization completed successfully, entering render loop\n");
    CompleteStartupLogging();
    overlay.Run();

    // 清理：允许系统重新进入休眠
    SetThreadExecutionState(ES_CONTINUOUS);

    printf("Exiting...\n");
    CoUninitialize();
    return 0;
}
