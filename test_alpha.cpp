/**
 * DX Overlay independent channel-strength telemetry test.
 *
 * Usage:
 *   test_alpha.exe [duration_seconds]
 *   test_alpha.exe [duration_seconds] [expected_both]
 *   test_alpha.exe [duration_seconds] [expected_static_cb] [expected_dynamic_cr]
 *
 * Values come from the same constant-buffer state consumed by the pixel
 * shader. The test also requires the lossless Y/Cr/Cb template format, because
 * legacy combined RGB templates cannot provide independent channel control.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "overlay_telemetry.h"

namespace {
struct MappingView {
    HANDLE mapping = nullptr;
    const DXOverlayTelemetry* telemetry = nullptr;

    ~MappingView() {
        if (telemetry) {
            UnmapViewOfFile(telemetry);
        }
        if (mapping) {
            CloseHandle(mapping);
        }
    }
};

bool OpenTelemetry(MappingView& view) {
    view.mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, kDXOverlayTelemetryMappingName);
    if (!view.mapping) {
        return false;
    }
    view.telemetry = static_cast<const DXOverlayTelemetry*>(MapViewOfFile(
        view.mapping, FILE_MAP_READ, 0, 0, sizeof(DXOverlayTelemetry)));
    return view.telemetry &&
           view.telemetry->magic == kDXOverlayTelemetryMagic &&
           view.telemetry->version == kDXOverlayTelemetryVersion &&
           view.telemetry->structSize == sizeof(DXOverlayTelemetry);
}

bool IsProcessAlive(DWORD processId) {
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, processId);
    if (!process) {
        return false;
    }
    const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
    CloseHandle(process);
    return alive;
}

LONG64 ReadPublished64(const volatile LONG64& value) {
    static_assert(sizeof(void*) == 8, "telemetry diagnostics require an x64 build");
    MemoryBarrier();
    const LONG64 result = value;
    MemoryBarrier();
    return result;
}

bool IsValidStrength(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}
}  // namespace

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    int durationSeconds = argc > 1 ? std::atoi(argv[1]) : 5;
    if (durationSeconds <= 0) {
        durationSeconds = 5;
    }

    const bool hasExpectedStrengths = argc > 2;
    const double expectedStatic = hasExpectedStrengths ? std::atof(argv[2]) : 0.0;
    const double expectedDynamic = argc > 3 ? std::atof(argv[3]) : expectedStatic;
    if (hasExpectedStrengths &&
        (!IsValidStrength(expectedStatic) || !IsValidStrength(expectedDynamic))) {
        std::printf("ERROR: expected strengths must be in [0, 1].\n");
        return 2;
    }

    MappingView view;
    if (!OpenTelemetry(view)) {
        std::printf("ERROR: DX Overlay telemetry is unavailable. Start dx_overlay.exe first.\n");
        return 1;
    }

    const DWORD processId = view.telemetry->processId;
    if (!view.telemetry->running || !IsProcessAlive(processId)) {
        std::printf("ERROR: telemetry exists but dx_overlay.exe is not running.\n");
        return 1;
    }

    const LONG64 initialPresents = ReadPublished64(view.telemetry->successfulPresents);
    const LONG64 initialConstantUpdates =
        ReadPublished64(view.telemetry->shaderConstantUpdates);
    LONG minStaticPpm = view.telemetry->staticAlphaPpm;
    LONG maxStaticPpm = minStaticPpm;
    LONG lastStaticPpm = minStaticPpm;
    LONG minDynamicPpm = view.telemetry->dynamicAlphaPpm;
    LONG maxDynamicPpm = minDynamicPpm;
    LONG lastDynamicPpm = minDynamicPpm;
    int staticChanges = 0;
    int dynamicChanges = 0;

    std::printf("DX Overlay independent channel-strength telemetry\n");
    std::printf("  PID: %lu\n", processId);
    std::printf("  Resolution: %ldx%ld @ %ld Hz\n",
        view.telemetry->screenWidth,
        view.telemetry->screenHeight,
        view.telemetry->refreshRate);
    std::printf("  Template format: %s\n",
        view.telemetry->channelEncodedTemplates
            ? "independent Y/Cr/Cb data"
            : "legacy combined RGB(A)");
    std::printf("  Initial static Cb strength: %.6f\n", lastStaticPpm / 1000000.0);
    std::printf("  Initial dynamic Cr strength: %.6f\n", lastDynamicPpm / 1000000.0);

    const ULONGLONG deadline =
        GetTickCount64() + static_cast<ULONGLONG>(durationSeconds) * 1000ULL;
    while (GetTickCount64() < deadline) {
        if (!view.telemetry->running || !IsProcessAlive(processId)) {
            std::printf("ERROR: overlay stopped during the test.\n");
            return 1;
        }

        MemoryBarrier();
        const LONG staticPpm = view.telemetry->staticAlphaPpm;
        const LONG dynamicPpm = view.telemetry->dynamicAlphaPpm;
        minStaticPpm = std::min(minStaticPpm, staticPpm);
        maxStaticPpm = std::max(maxStaticPpm, staticPpm);
        minDynamicPpm = std::min(minDynamicPpm, dynamicPpm);
        maxDynamicPpm = std::max(maxDynamicPpm, dynamicPpm);
        if (staticPpm != lastStaticPpm) {
            ++staticChanges;
            std::printf("  Static Cb strength changed: %.6f -> %.6f\n",
                lastStaticPpm / 1000000.0,
                staticPpm / 1000000.0);
            lastStaticPpm = staticPpm;
        }
        if (dynamicPpm != lastDynamicPpm) {
            ++dynamicChanges;
            std::printf("  Dynamic Cr strength changed: %.6f -> %.6f\n",
                lastDynamicPpm / 1000000.0,
                dynamicPpm / 1000000.0);
            lastDynamicPpm = dynamicPpm;
        }
        Sleep(50);
    }

    const LONG64 presentedFrames =
        ReadPublished64(view.telemetry->successfulPresents) - initialPresents;
    const LONG64 constantUpdates =
        ReadPublished64(view.telemetry->shaderConstantUpdates) - initialConstantUpdates;
    const double observedStatic = lastStaticPpm / 1000000.0;
    const double observedDynamic = lastDynamicPpm / 1000000.0;
    const bool staticMatches = !hasExpectedStrengths ||
        std::fabs(observedStatic - expectedStatic) <= 0.000001;
    const bool dynamicMatches = !hasExpectedStrengths ||
        std::fabs(observedDynamic - expectedDynamic) <= 0.000001;
    const bool passed = view.telemetry->channelEncodedTemplates &&
        staticChanges == 0 && dynamicChanges == 0 &&
        constantUpdates > 0 && presentedFrames > 0 &&
        staticMatches && dynamicMatches;

    std::printf("\nResult\n");
    std::printf("  Static Cb range: %.6f .. %.6f\n",
        minStaticPpm / 1000000.0,
        maxStaticPpm / 1000000.0);
    std::printf("  Dynamic Cr range: %.6f .. %.6f\n",
        minDynamicPpm / 1000000.0,
        maxDynamicPpm / 1000000.0);
    std::printf("  Strength changes (static/dynamic): %d/%d\n",
        staticChanges,
        dynamicChanges);
    std::printf("  Shader constant writes observed: %lld\n",
        static_cast<long long>(constantUpdates));
    std::printf("  Successful presents observed: %lld\n",
        static_cast<long long>(presentedFrames));
    if (hasExpectedStrengths) {
        std::printf("  Expected static Cb: %.6f (%s)\n",
            expectedStatic,
            staticMatches ? "matched" : "mismatch");
        std::printf("  Expected dynamic Cr: %.6f (%s)\n",
            expectedDynamic,
            dynamicMatches ? "matched" : "mismatch");
    }
    std::printf("  STATUS: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
