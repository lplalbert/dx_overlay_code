#pragma once

#include "overlay.h"

#include <avrt.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>

extern const char* g_vertexShaderCode;
extern const char* g_pixelShaderCode;

void LogToFile(const char* msg);

inline constexpr UINT kSwapChainBufferCount = 2;
inline constexpr UINT kMpoSwapChainBufferCount = 2;
inline constexpr UINT kSwapChainResizeFlags =
    DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
inline constexpr DWORD kFrameLatencyWaitTimeoutMs = 100;
inline constexpr int kMaxFrameLatencyTimeoutsBeforeRecovery = 5;
inline constexpr int kMaxFrameLatencyWaitFailuresBeforeRecovery = 3;
inline constexpr bool kEnableMPO = false;

struct AtomicFlagScope {
    std::atomic<bool>& flag;
    explicit AtomicFlagScope(std::atomic<bool>& value) : flag(value) {}
    ~AtomicFlagScope() { flag.store(false); }
};

struct MmcssRegistration {
    using SetCharacteristicsFn = HANDLE(WINAPI*)(LPCWSTR, LPDWORD);
    using SetPriorityFn = BOOL(WINAPI*)(HANDLE, AVRT_PRIORITY);
    using RevertFn = BOOL(WINAPI*)(HANDLE);

    HMODULE module = nullptr;
    HANDLE handle = nullptr;
    SetCharacteristicsFn setCharacteristics = nullptr;
    SetPriorityFn setPriority = nullptr;
    RevertFn revert = nullptr;
    std::wstring profile;

    bool Activate(const wchar_t* profileName, AVRT_PRIORITY priority, DWORD* lastError = nullptr) {
        Reset();
        module = LoadLibraryW(L"avrt.dll");
        if (!module) {
            if (lastError) *lastError = GetLastError();
            return false;
        }
        setCharacteristics = reinterpret_cast<SetCharacteristicsFn>(
            GetProcAddress(module, "AvSetMmThreadCharacteristicsW"));
        setPriority = reinterpret_cast<SetPriorityFn>(
            GetProcAddress(module, "AvSetMmThreadPriority"));
        revert = reinterpret_cast<RevertFn>(
            GetProcAddress(module, "AvRevertMmThreadCharacteristics"));
        if (!setCharacteristics || !revert) {
            if (lastError) *lastError = ERROR_PROC_NOT_FOUND;
            Reset();
            return false;
        }
        DWORD taskIndex = 0;
        handle = setCharacteristics(profileName, &taskIndex);
        if (!handle) {
            if (lastError) *lastError = GetLastError();
            Reset();
            return false;
        }
        profile = profileName;
        if (setPriority) setPriority(handle, priority);
        if (lastError) *lastError = ERROR_SUCCESS;
        return true;
    }

    void Reset() {
        if (handle && revert) revert(handle);
        handle = nullptr;
        setCharacteristics = nullptr;
        setPriority = nullptr;
        revert = nullptr;
        profile.clear();
        if (module) {
            FreeLibrary(module);
            module = nullptr;
        }
    }

    ~MmcssRegistration() { Reset(); }
};
