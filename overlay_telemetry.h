#pragma once

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstddef>
#include <cstdint>

inline constexpr wchar_t kDXOverlayTelemetryMappingName[] =
    L"Local\\DXOverlay.Telemetry.v5";
inline constexpr std::uint32_t kDXOverlayTelemetryMagic = 0x44584F54u;
inline constexpr std::uint32_t kDXOverlayTelemetryVersion = 5u;
inline constexpr std::size_t kDXOverlayPresentHistoryCapacity = 512u;

struct alignas(32) DXOverlayDisplayFrameSample {
    volatile LONG64 presentCount = 0;
    volatile LONG64 presentRefreshCount = 0;
    volatile LONG64 syncRefreshCount = 0;
    volatile LONG64 syncQpc = 0;
    volatile LONG textureIndex = -1;
    volatile LONG reserved = 0;
};

// Cross-process runtime telemetry for the diagnostic executables. The overlay
// is the only writer. Each ring sequence is published only after all fields in
// that sample have been stored, so readers can take the sequence as the commit
// marker for successful-Present and displayed-frame histories.
struct alignas(64) DXOverlayTelemetry {
    std::uint32_t magic = kDXOverlayTelemetryMagic;
    std::uint32_t version = kDXOverlayTelemetryVersion;
    std::uint32_t structSize = 0;
    std::uint32_t processId = 0;

    volatile LONG running = 0;
    volatile LONG shaderAlphaPpm = 0;  // max(staticAlpha, dynamicAlpha), retained for compatibility
    volatile LONG staticAlphaPpm = 0;
    volatile LONG dynamicAlphaPpm = 0;
    volatile LONG currentTextureIndex = 0;
    volatile LONG textureCount = 0;
    volatile LONG screenWidth = 0;
    volatile LONG screenHeight = 0;
    volatile LONG refreshRate = 0;
    volatile LONG dynamicUsesSourceAlpha = 0;
    volatile LONG channelEncodedTemplates = 0;
    volatile LONG holdFrames = 1;
    volatile LONG transitionFrames = 1;
    volatile LONG64 shaderConstantUpdates = 0;

    LARGE_INTEGER qpcFrequency = {};
    volatile LONG64 successfulPresents = 0;
    volatile LONG64 failedPresents = 0;
    volatile LONG64 lastPresentQpc = 0;
    volatile LONG lastPresentHresult = static_cast<LONG>(S_OK);
    volatile LONG lastFrameStatisticsHresult = static_cast<LONG>(S_OK);
    volatile LONG64 frameStatisticsSamples = 0;
    volatile LONG64 frameStatisticsFailures = 0;
    volatile LONG64 frameStatisticsDisjoint = 0;
    volatile LONG64 phaseRefreshCorrection = 0;
    volatile LONG64 phaseResynchronizations = 0;
    volatile LONG64 phaseCorrectionSteps = 0;
    volatile LONG64 lastDisplayedPresentCount = 0;
    volatile LONG64 lastPresentRefreshCount = 0;
    volatile LONG64 lastSyncRefreshCount = 0;
    volatile LONG64 lastSyncQpc = 0;
    volatile LONG64 presentQpcHistory[kDXOverlayPresentHistoryCapacity] = {};
    volatile LONG64 presentIdHistory[kDXOverlayPresentHistoryCapacity] = {};
    volatile LONG64 presentPhaseOrdinalHistory[kDXOverlayPresentHistoryCapacity] = {};
    volatile LONG presentTextureIndexHistory[kDXOverlayPresentHistoryCapacity] = {};
    DXOverlayDisplayFrameSample displayFrameHistory[kDXOverlayPresentHistoryCapacity] = {};
};
