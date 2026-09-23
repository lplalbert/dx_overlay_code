/**
 * DX Overlay submission/display synchronization test.
 *
 * Usage: test_frame3.exe [duration_seconds]
 *
 * The test reports two independent timelines:
 *   1. successful Present submissions, timestamped by the overlay;
 *   2. frames DXGI reports as actually displayed, timestamped at vblank.
 *
 * A successful Present is not treated as proof that the frame reached the
 * display. If DXGI display statistics are unavailable, the result is
 * INCONCLUSIVE instead of a misleading pass.
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

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

struct DisplaySample {
    std::uint32_t presentCount = 0;
    std::uint32_t presentRefreshCount = 0;
    std::uint32_t syncRefreshCount = 0;
    LONG64 syncQpc = 0;
    int textureIndex = -1;
};

struct TimingSummary {
    std::vector<double> intervalsMs;
    double fps = 0.0;
    double averageMs = 0.0;
    double jitterMs = 0.0;
    int longIntervals = 0;
};

void Print(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsprintf_s(buffer, format, args);
    va_end(args);

    std::fputs(buffer, stdout);
    FILE* file = std::fopen("frame_test_log.txt", "a");
    if (file) {
        std::fputs(buffer, file);
        std::fclose(file);
    }
}

LONG64 ReadPublished64(const volatile LONG64& value) {
    // The telemetry mapping is read-only. The release target is x64, where
    // aligned 64-bit loads are atomic; barriers preserve publication order.
    static_assert(sizeof(void*) == 8, "telemetry diagnostics require an x64 build");
    MemoryBarrier();
    const LONG64 result = value;
    MemoryBarrier();
    return result;
}

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

bool IsProcessAlive(HANDLE process) {
    return process && WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
}

double Percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percentile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

TimingSummary AnalyzeTiming(
    const std::vector<LONG64>& timestamps,
    double qpcFrequency,
    double targetMs) {
    TimingSummary summary;
    if (timestamps.size() < 2 || timestamps.back() <= timestamps.front()) {
        return summary;
    }

    summary.intervalsMs.reserve(timestamps.size() - 1);
    for (std::size_t i = 1; i < timestamps.size(); ++i) {
        summary.intervalsMs.push_back(
            static_cast<double>(timestamps[i] - timestamps[i - 1]) *
            1000.0 / qpcFrequency);
    }
    summary.averageMs = std::accumulate(
        summary.intervalsMs.begin(),
        summary.intervalsMs.end(),
        0.0) / static_cast<double>(summary.intervalsMs.size());

    double squaredError = 0.0;
    for (double interval : summary.intervalsMs) {
        const double error = interval - summary.averageMs;
        squaredError += error * error;
        if (interval > targetMs * 1.5) {
            ++summary.longIntervals;
        }
    }
    summary.jitterMs = std::sqrt(
        squaredError / static_cast<double>(summary.intervalsMs.size()));
    summary.fps = qpcFrequency * static_cast<double>(timestamps.size() - 1) /
        static_cast<double>(timestamps.back() - timestamps.front());
    return summary;
}

void PrintTimingSummary(
    const char* title,
    const TimingSummary& summary,
    std::size_t sampleCount) {
    Print("%s\n", title);
    Print("  Samples: %zu\n", sampleCount);
    Print("  Measured FPS: %.2f\n", summary.fps);
    Print("  Average interval: %.3f ms\n", summary.averageMs);
    Print("  P50 / P95 / P99: %.3f / %.3f / %.3f ms\n",
        Percentile(summary.intervalsMs, 0.50),
        Percentile(summary.intervalsMs, 0.95),
        Percentile(summary.intervalsMs, 0.99));
    Print("  Min / Max: %.3f / %.3f ms\n",
        *std::min_element(summary.intervalsMs.begin(), summary.intervalsMs.end()),
        *std::max_element(summary.intervalsMs.begin(), summary.intervalsMs.end()));
    Print("  Jitter (stddev): %.3f ms\n", summary.jitterMs);
    Print("  Long intervals (>1.5x): %d / %zu (%.2f%%)\n",
        summary.longIntervals,
        summary.intervalsMs.size(),
        100.0 * static_cast<double>(summary.longIntervals) /
            static_cast<double>(summary.intervalsMs.size()));
}

std::uint64_t CounterDelta(std::uint32_t current, std::uint32_t previous) {
    if (current >= previous) {
        return static_cast<std::uint64_t>(current - previous);
    }
    return (std::uint64_t{1} << 32) - previous + current;
}
}  // namespace

int main(int argc, char* argv[]) {
    DeleteFileA("frame_test_log.txt");
    setvbuf(stdout, nullptr, _IONBF, 0);

    int durationSeconds = argc > 1 ? std::atoi(argv[1]) : 10;
    if (durationSeconds <= 0) {
        durationSeconds = 10;
    }

    MappingView view;
    if (!OpenTelemetry(view)) {
        Print("ERROR: DX Overlay telemetry is unavailable. Start dx_overlay.exe first.\n");
        return 1;
    }

    const DWORD processId = view.telemetry->processId;
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, processId);
    if (!view.telemetry->running || !IsProcessAlive(process)) {
        if (process) {
            CloseHandle(process);
        }
        Print("ERROR: telemetry exists but dx_overlay.exe is not running.\n");
        return 1;
    }

    const LONG publishedRefreshRate = view.telemetry->refreshRate;
    const LONG refreshRate = std::max<LONG>(1, publishedRefreshRate);
    const double targetMs = 1000.0 / static_cast<double>(refreshRate);
    const double qpcFrequency =
        static_cast<double>(view.telemetry->qpcFrequency.QuadPart);
    if (qpcFrequency <= 0.0) {
        CloseHandle(process);
        Print("ERROR: telemetry contains an invalid QPC frequency.\n");
        return 1;
    }

    const LONG textureCount = view.telemetry->textureCount;
    const LONG holdFrames = view.telemetry->holdFrames;
    const LONG transitionFrames = view.telemetry->transitionFrames;
    const bool hardAlternation =
        textureCount >= 2 && holdFrames == 1 && transitionFrames == 1;

    Print("DX Overlay submission/display synchronization test\n");
    Print("  PID: %lu\n", processId);
    Print("  Display: %ldx%ld @ %ld Hz\n",
        view.telemetry->screenWidth,
        view.telemetry->screenHeight,
        refreshRate);
    Print("  Texture phase: hold=%ld, transition=%ld, textures=%ld\n",
        holdFrames,
        transitionFrames,
        textureCount);
    Print("  Target interval: %.3f ms\n", targetMs);
    Print("  Duration: %d seconds\n\n", durationSeconds);

    LONG64 consumedSubmission = ReadPublished64(view.telemetry->successfulPresents);
    LONG64 consumedDisplay = ReadPublished64(view.telemetry->frameStatisticsSamples);
    const LONG64 initialPresentFailures = ReadPublished64(view.telemetry->failedPresents);
    const LONG64 initialStatisticsFailures =
        ReadPublished64(view.telemetry->frameStatisticsFailures);
    const LONG64 initialStatisticsDisjoint =
        ReadPublished64(view.telemetry->frameStatisticsDisjoint);
    const LONG64 initialPhaseResynchronizations =
        ReadPublished64(view.telemetry->phaseResynchronizations);
    const LONG64 initialPhaseCorrectionSteps =
        ReadPublished64(view.telemetry->phaseCorrectionSteps);
    LONG64 lostSubmissionSamples = 0;
    LONG64 lostDisplaySamples = 0;
    std::vector<LONG64> submissionTimestamps;
    std::vector<DisplaySample> displaySamples;

    const ULONGLONG deadline = GetTickCount64() +
        static_cast<ULONGLONG>(durationSeconds) * 1000ULL;
    while (GetTickCount64() < deadline) {
        if (!view.telemetry->running || !IsProcessAlive(process)) {
            CloseHandle(process);
            Print("ERROR: overlay stopped during the test.\n");
            return 1;
        }

        const LONG64 publishedSubmission =
            ReadPublished64(view.telemetry->successfulPresents);
        if (publishedSubmission - consumedSubmission >
            static_cast<LONG64>(kDXOverlayPresentHistoryCapacity)) {
            const LONG64 recoverableStart = publishedSubmission -
                static_cast<LONG64>(kDXOverlayPresentHistoryCapacity) + 1;
            lostSubmissionSamples += recoverableStart - (consumedSubmission + 1);
            consumedSubmission = recoverableStart - 1;
        }
        while (consumedSubmission < publishedSubmission) {
            const LONG64 sequence = consumedSubmission + 1;
            const std::size_t index = static_cast<std::size_t>(sequence) %
                kDXOverlayPresentHistoryCapacity;
            const LONG64 timestamp =
                ReadPublished64(view.telemetry->presentQpcHistory[index]);
            if (timestamp > 0 &&
                (submissionTimestamps.empty() ||
                 timestamp > submissionTimestamps.back())) {
                submissionTimestamps.push_back(timestamp);
            } else {
                ++lostSubmissionSamples;
            }
            consumedSubmission = sequence;
        }

        const LONG64 publishedDisplay =
            ReadPublished64(view.telemetry->frameStatisticsSamples);
        if (publishedDisplay - consumedDisplay >
            static_cast<LONG64>(kDXOverlayPresentHistoryCapacity)) {
            const LONG64 recoverableStart = publishedDisplay -
                static_cast<LONG64>(kDXOverlayPresentHistoryCapacity) + 1;
            lostDisplaySamples += recoverableStart - (consumedDisplay + 1);
            consumedDisplay = recoverableStart - 1;
        }
        while (consumedDisplay < publishedDisplay) {
            const LONG64 sequence = consumedDisplay + 1;
            const std::size_t index = static_cast<std::size_t>(sequence) %
                kDXOverlayPresentHistoryCapacity;
            const DXOverlayDisplayFrameSample& published =
                view.telemetry->displayFrameHistory[index];
            DisplaySample sample;
            sample.presentCount = static_cast<std::uint32_t>(
                ReadPublished64(published.presentCount));
            sample.presentRefreshCount = static_cast<std::uint32_t>(
                ReadPublished64(published.presentRefreshCount));
            sample.syncRefreshCount = static_cast<std::uint32_t>(
                ReadPublished64(published.syncRefreshCount));
            sample.syncQpc = ReadPublished64(published.syncQpc);
            sample.textureIndex = published.textureIndex;
            if (sample.syncQpc > 0 &&
                (displaySamples.empty() ||
                 sample.syncQpc > displaySamples.back().syncQpc)) {
                displaySamples.push_back(sample);
            } else {
                ++lostDisplaySamples;
            }
            consumedDisplay = sequence;
        }
        Sleep(1);
    }

    const LONG64 failedPresents =
        ReadPublished64(view.telemetry->failedPresents) - initialPresentFailures;
    const LONG64 statisticsFailures =
        ReadPublished64(view.telemetry->frameStatisticsFailures) -
        initialStatisticsFailures;
    const LONG64 statisticsDisjoint =
        ReadPublished64(view.telemetry->frameStatisticsDisjoint) -
        initialStatisticsDisjoint;
    const LONG64 phaseResynchronizations =
        ReadPublished64(view.telemetry->phaseResynchronizations) -
        initialPhaseResynchronizations;
    const LONG64 phaseCorrectionSteps =
        ReadPublished64(view.telemetry->phaseCorrectionSteps) -
        initialPhaseCorrectionSteps;
    const LONG64 finalPhaseCorrection =
        ReadPublished64(view.telemetry->phaseRefreshCorrection);

    if (submissionTimestamps.size() < 2) {
        CloseHandle(process);
        Print("ERROR: fewer than two successful Present submissions were observed.\n");
        return 1;
    }

    const TimingSummary submissionTiming = AnalyzeTiming(
        submissionTimestamps,
        qpcFrequency,
        targetMs);
    PrintTimingSummary(
        "Present submission timeline",
        submissionTiming,
        submissionTimestamps.size());
    Print("  Failed Present calls: %lld\n",
        static_cast<long long>(failedPresents));
    Print("  Lost submission telemetry: %lld\n\n",
        static_cast<long long>(lostSubmissionSamples));

    const double submissionLongRatio =
        static_cast<double>(submissionTiming.longIntervals) /
        static_cast<double>(submissionTiming.intervalsMs.size());
    const bool submissionPassed =
        failedPresents == 0 &&
        lostSubmissionSamples == 0 &&
        submissionTiming.fps >= static_cast<double>(refreshRate) * 0.90 &&
        submissionLongRatio <= 0.01;

    if (displaySamples.size() < 2) {
        Print("Displayed-frame timeline\n");
        Print("  DXGI display samples: %zu\n", displaySamples.size());
        Print("  Statistics failures: %lld\n",
            static_cast<long long>(statisticsFailures));
        Print("  Statistics disjoint events: %lld\n",
            static_cast<long long>(statisticsDisjoint));
        Print("  Last statistics HRESULT: 0x%08lX\n",
            static_cast<unsigned long>(view.telemetry->lastFrameStatisticsHresult));
        Print("\nSTATUS: INCONCLUSIVE - Present submissions were measured, but "
              "DXGI did not provide enough displayed-frame samples.\n");
        CloseHandle(process);
        return 2;
    }

    std::vector<LONG64> displayTimestamps;
    displayTimestamps.reserve(displaySamples.size());
    for (const DisplaySample& sample : displaySamples) {
        // SyncQPCTime belongs to SyncRefreshCount. Move that QPC timestamp to
        // the vblank identified by PresentRefreshCount to estimate when this
        // particular Present ID actually reached the display.
        const std::uint64_t refreshLead = CounterDelta(
            sample.presentRefreshCount,
            sample.syncRefreshCount);
        const double displayedQpc = static_cast<double>(sample.syncQpc) +
            static_cast<double>(refreshLead) * qpcFrequency /
                static_cast<double>(refreshRate);
        displayTimestamps.push_back(static_cast<LONG64>(std::llround(displayedQpc)));
    }
    const TimingSummary displayTiming = AnalyzeTiming(
        displayTimestamps,
        qpcFrequency,
        targetMs);

    std::uint64_t observedRefreshIntervals = 0;
    std::uint64_t skippedRefreshes = 0;
    std::uint64_t droppedSubmissions = 0;
    std::uint64_t observedPresentAdvances = 0;
    std::uint64_t invalidRefreshDeltas = 0;
    std::uint64_t unmatchedTextureSamples = 0;
    std::uint64_t phaseClockDiscontinuities = 0;
    std::uint64_t hardPhaseComparisons = 0;
    std::uint64_t hardPhaseMismatches = 0;
    std::uint64_t unexplainedPhaseMismatches = 0;
    for (const DisplaySample& sample : displaySamples) {
        if (sample.textureIndex < 0) {
            ++unmatchedTextureSamples;
        }
    }
    for (std::size_t i = 1; i < displaySamples.size(); ++i) {
        const std::uint64_t refreshDelta = CounterDelta(
            displaySamples[i].presentRefreshCount,
            displaySamples[i - 1].presentRefreshCount);
        const std::uint64_t presentDelta = CounterDelta(
            displaySamples[i].presentCount,
            displaySamples[i - 1].presentCount);
        observedPresentAdvances += presentDelta;
        if (refreshDelta == 0) {
            ++invalidRefreshDeltas;
        } else {
            observedRefreshIntervals += refreshDelta;
            skippedRefreshes += refreshDelta - 1;
        }
        if (presentDelta > 1) {
            droppedSubmissions += presentDelta - 1;
        }
        if (refreshDelta > 0 && presentDelta > 0 &&
            refreshDelta != presentDelta) {
            ++phaseClockDiscontinuities;
        }

        if (hardAlternation && refreshDelta > 0 &&
            displaySamples[i - 1].textureIndex >= 0 &&
            displaySamples[i].textureIndex >= 0) {
            const bool shouldMatch = (refreshDelta % 2) == 0;
            const bool doesMatch =
                displaySamples[i - 1].textureIndex ==
                displaySamples[i].textureIndex;
            ++hardPhaseComparisons;
            if (shouldMatch != doesMatch) {
                ++hardPhaseMismatches;
                if (refreshDelta == presentDelta) {
                    ++unexplainedPhaseMismatches;
                }
            }
        }
    }

    PrintTimingSummary(
        "Displayed-frame timeline (DXGI present-vblank estimate)",
        displayTiming,
        displaySamples.size());
    Print("  Lost display telemetry: %lld\n",
        static_cast<long long>(lostDisplaySamples));
    Print("  Statistics failures / disjoint: %lld / %lld\n",
        static_cast<long long>(statisticsFailures),
        static_cast<long long>(statisticsDisjoint));
    Print("  Skipped refreshes: %llu / %llu (%.2f%%)\n",
        static_cast<unsigned long long>(skippedRefreshes),
        static_cast<unsigned long long>(observedRefreshIntervals),
        observedRefreshIntervals > 0
            ? 100.0 * static_cast<double>(skippedRefreshes) /
                static_cast<double>(observedRefreshIntervals)
            : 100.0);
    Print("  Submitted frames not observed as displayed: %llu\n",
        static_cast<unsigned long long>(droppedSubmissions));
    Print("  Invalid refresh deltas: %llu\n",
        static_cast<unsigned long long>(invalidRefreshDeltas));
    Print("  Displayed Present IDs without texture match: %llu\n",
        static_cast<unsigned long long>(unmatchedTextureSamples));
    Print("  Physical-clock discontinuities: %llu\n",
        static_cast<unsigned long long>(phaseClockDiscontinuities));
    Print("  Phase resynchronizations / corrected steps: %lld / %lld\n",
        static_cast<long long>(phaseResynchronizations),
        static_cast<long long>(phaseCorrectionSteps));
    Print("  Final phase correction offset: %lld\n",
        static_cast<long long>(finalPhaseCorrection));
    if (hardAlternation) {
        Print("  Hard-alternation phase mismatches: %llu / %llu\n",
            static_cast<unsigned long long>(hardPhaseMismatches),
            static_cast<unsigned long long>(hardPhaseComparisons));
        Print("  Unexplained phase mismatches after resync: %llu\n",
            static_cast<unsigned long long>(unexplainedPhaseMismatches));
    } else {
        Print("  Hard-alternation phase check: not applicable\n");
    }

    const double displayLongRatio =
        static_cast<double>(displayTiming.longIntervals) /
        static_cast<double>(displayTiming.intervalsMs.size());
    const double skippedRefreshRatio = observedRefreshIntervals > 0
        ? static_cast<double>(skippedRefreshes) /
            static_cast<double>(observedRefreshIntervals)
        : 1.0;
    const double unobservedPresentRatio = observedPresentAdvances > 0
        ? static_cast<double>(droppedSubmissions) /
            static_cast<double>(observedPresentAdvances)
        : 1.0;
    const bool resynchronizationObserved =
        phaseClockDiscontinuities == 0 || phaseResynchronizations > 0;
    const bool phasePassed =
        !hardAlternation ||
        (hardPhaseComparisons > 0 &&
         unexplainedPhaseMismatches == 0 &&
         resynchronizationObserved);
    const bool displayPassed =
        lostDisplaySamples == 0 &&
        statisticsFailures == 0 &&
        statisticsDisjoint == 0 &&
        displayTiming.fps >= static_cast<double>(refreshRate) * 0.90 &&
        displayLongRatio <= 0.01 &&
        skippedRefreshRatio <= 0.01 &&
        unobservedPresentRatio <= 0.01 &&
        invalidRefreshDeltas == 0 &&
        unmatchedTextureSamples == 0 &&
        phasePassed;
    const bool passed = submissionPassed && displayPassed;
    const bool recoveredTransient = passed &&
        (phaseClockDiscontinuities > 0 ||
         skippedRefreshes > 0 ||
         droppedSubmissions > 0 ||
         hardPhaseMismatches > 0);
    Print("\nSTATUS: %s\n",
        passed
            ? (recoveredTransient ? "PASS_WITH_PHASE_RECOVERY" : "PASS")
            : "FAIL");

    CloseHandle(process);
    return passed ? 0 : 1;
}
