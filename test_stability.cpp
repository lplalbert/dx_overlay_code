/**
 * dx_overlay 稳定性测试工具
 * 
 * 监控指标：
 * 1. 进程存活状态
 * 2. CPU 使用率
 * 3. 内存使用量（工作集）
 * 4. GDI 句柄数
 * 5. 线程数
 * 6. 页面错误数（可检测内存泄漏）
 * 
 * 编译: g++ -std=c++17 -O2 -o test_stability.exe test_stability.cpp -lpsapi
 * 运行: test_stability.exe [监控时长秒数，默认60]
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <numeric>

// 采样数据结构
struct SampleData {
    double timestamp;       // 相对时间（秒）
    double cpuUsage;        // CPU 使用率 (%)
    SIZE_T workingSetMB;    // 工作集内存 (MB)
    SIZE_T privateBytesMB;  // 私有内存 (MB)
    DWORD gdiHandles;       // GDI 句柄数
    DWORD userHandles;      // USER 句柄数
    DWORD threadCount;      // 线程数
    DWORD pageFaults;       // 页面错误总数
    bool isRunning;         // 进程是否存活
};

// 统计结果
struct Statistics {
    double min;
    double max;
    double avg;
    double stdDev;
    double trend;  // 趋势斜率（正值表示上升）
};

class StabilityMonitor {
public:
    StabilityMonitor(const std::wstring& processName, int durationSec, int intervalMs = 500)
        : m_processName(processName)
        , m_durationSec(durationSec)
        , m_intervalMs(intervalMs)
    {
        // 获取 CPU 核心数用于计算 CPU 使用率
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        m_numProcessors = sysInfo.dwNumberOfProcessors;
    }

    bool Run() {
        std::wcout << L"========================================\n";
        std::wcout << L"  dx_overlay 稳定性监控测试\n";
        std::wcout << L"========================================\n";
        std::wcout << L"目标进程: " << m_processName << L"\n";
        std::wcout << L"监控时长: " << m_durationSec << L" 秒\n";
        std::wcout << L"采样间隔: " << m_intervalMs << L" ms\n";
        std::wcout << L"========================================\n\n";

        // 查找进程
        m_processId = FindProcessId(m_processName);
        if (m_processId == 0) {
            std::wcerr << L"错误: 未找到进程 " << m_processName << L"\n";
            std::wcerr << L"请先启动 dx_overlay.exe\n";
            return false;
        }

        std::wcout << L"找到进程 PID: " << m_processId << L"\n\n";

        // 打开进程
        m_hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, m_processId);
        if (!m_hProcess) {
            std::wcerr << L"错误: 无法打开进程，错误码: " << GetLastError() << L"\n";
            return false;
        }

        // 初始化 CPU 时间
        FILETIME createTime, exitTime, kernelTime, userTime;
        GetProcessTimes(m_hProcess, &createTime, &exitTime, &kernelTime, &userTime);
        m_lastKernelTime = FileTimeToUInt64(kernelTime);
        m_lastUserTime = FileTimeToUInt64(userTime);
        m_lastCheckTime = std::chrono::high_resolution_clock::now();

        // 打印表头
        PrintHeader();

        // 开始监控
        auto startTime = std::chrono::high_resolution_clock::now();
        int sampleCount = 0;
        int crashCount = 0;

        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();

            if (elapsed >= m_durationSec) break;

            SampleData sample = CollectSample(elapsed);
            m_samples.push_back(sample);

            PrintSample(sample, sampleCount);
            sampleCount++;

            if (!sample.isRunning) {
                crashCount++;
                std::wcout << L"\n*** 警告: 进程已停止运行! ***\n";
                
                // 尝试重新查找（可能是重启了）
                Sleep(1000);
                m_processId = FindProcessId(m_processName);
                if (m_processId != 0) {
                    CloseHandle(m_hProcess);
                    m_hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, m_processId);
                    if (m_hProcess) {
                        std::wcout << L"进程已重启，继续监控...\n";
                        // 重置 CPU 时间
                        GetProcessTimes(m_hProcess, &createTime, &exitTime, &kernelTime, &userTime);
                        m_lastKernelTime = FileTimeToUInt64(kernelTime);
                        m_lastUserTime = FileTimeToUInt64(userTime);
                        m_lastCheckTime = std::chrono::high_resolution_clock::now();
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(m_intervalMs));
        }

        CloseHandle(m_hProcess);

        // 生成报告
        std::wcout << L"\n\n";
        GenerateReport(crashCount);

        // 保存到文件
        SaveToCSV();

        return true;
    }

private:
    DWORD FindProcessId(const std::wstring& processName) {
        DWORD pid = 0;
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W pe;
            pe.dwSize = sizeof(pe);
            if (Process32FirstW(snapshot, &pe)) {
                do {
                    if (processName == pe.szExeFile) {
                        pid = pe.th32ProcessID;
                        break;
                    }
                } while (Process32NextW(snapshot, &pe));
            }
            CloseHandle(snapshot);
        }
        return pid;
    }

    DWORD GetThreadCount(DWORD processId) {
        DWORD count = 0;
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot != INVALID_HANDLE_VALUE) {
            THREADENTRY32 te;
            te.dwSize = sizeof(te);
            if (Thread32First(snapshot, &te)) {
                do {
                    if (te.th32OwnerProcessID == processId) {
                        count++;
                    }
                } while (Thread32Next(snapshot, &te));
            }
            CloseHandle(snapshot);
        }
        return count;
    }

    UINT64 FileTimeToUInt64(const FILETIME& ft) {
        return (static_cast<UINT64>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    }

    SampleData CollectSample(double timestamp) {
        SampleData sample = {};
        sample.timestamp = timestamp;
        sample.isRunning = true;

        // 检查进程是否还在运行
        DWORD exitCode;
        if (!GetExitCodeProcess(m_hProcess, &exitCode) || exitCode != STILL_ACTIVE) {
            sample.isRunning = false;
            return sample;
        }

        // 内存信息
        PROCESS_MEMORY_COUNTERS_EX pmc;
        pmc.cb = sizeof(pmc);
        if (GetProcessMemoryInfo(m_hProcess, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            sample.workingSetMB = pmc.WorkingSetSize / (1024 * 1024);
            sample.privateBytesMB = pmc.PrivateUsage / (1024 * 1024);
            sample.pageFaults = pmc.PageFaultCount;
        }

        // GDI 和 USER 句柄
        sample.gdiHandles = GetGuiResources(m_hProcess, GR_GDIOBJECTS);
        sample.userHandles = GetGuiResources(m_hProcess, GR_USEROBJECTS);

        // 线程数
        sample.threadCount = GetThreadCount(m_processId);

        // CPU 使用率
        FILETIME createTime, exitTime, kernelTime, userTime;
        if (GetProcessTimes(m_hProcess, &createTime, &exitTime, &kernelTime, &userTime)) {
            UINT64 kernelTimeNow = FileTimeToUInt64(kernelTime);
            UINT64 userTimeNow = FileTimeToUInt64(userTime);
            
            auto now = std::chrono::high_resolution_clock::now();
            double elapsedSec = std::chrono::duration<double>(now - m_lastCheckTime).count();
            
            if (elapsedSec > 0) {
                UINT64 cpuTimeDelta = (kernelTimeNow - m_lastKernelTime) + (userTimeNow - m_lastUserTime);
                // FILETIME 单位是 100 纳秒
                double cpuTimeSec = cpuTimeDelta / 10000000.0;
                sample.cpuUsage = (cpuTimeSec / elapsedSec) * 100.0 / m_numProcessors;
            }
            
            m_lastKernelTime = kernelTimeNow;
            m_lastUserTime = userTimeNow;
            m_lastCheckTime = now;
        }

        return sample;
    }

    void PrintHeader() {
        std::wcout << std::setw(8) << L"时间(s)"
                   << std::setw(10) << L"CPU(%)"
                   << std::setw(12) << L"内存(MB)"
                   << std::setw(12) << L"私有(MB)"
                   << std::setw(8) << L"GDI"
                   << std::setw(8) << L"USER"
                   << std::setw(8) << L"线程"
                   << std::setw(12) << L"页错误"
                   << std::setw(8) << L"状态"
                   << L"\n";
        std::wcout << std::wstring(86, L'-') << L"\n";
    }

    void PrintSample(const SampleData& sample, int index) {
        // 每 20 行重新打印表头
        if (index > 0 && index % 20 == 0) {
            std::wcout << L"\n";
            PrintHeader();
        }

        std::wcout << std::fixed << std::setprecision(1)
                   << std::setw(8) << sample.timestamp
                   << std::setw(10) << sample.cpuUsage
                   << std::setw(12) << sample.workingSetMB
                   << std::setw(12) << sample.privateBytesMB
                   << std::setw(8) << sample.gdiHandles
                   << std::setw(8) << sample.userHandles
                   << std::setw(8) << sample.threadCount
                   << std::setw(12) << sample.pageFaults
                   << std::setw(8) << (sample.isRunning ? L"运行" : L"停止")
                   << L"\n";
    }

    Statistics CalculateStats(const std::vector<double>& values) {
        Statistics stats = {};
        if (values.empty()) return stats;

        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());
        stats.avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

        // 标准差
        double sumSq = 0;
        for (double v : values) {
            sumSq += (v - stats.avg) * (v - stats.avg);
        }
        stats.stdDev = std::sqrt(sumSq / values.size());

        // 线性回归计算趋势
        if (values.size() >= 2) {
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (size_t i = 0; i < values.size(); i++) {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }
            double n = static_cast<double>(values.size());
            stats.trend = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        }

        return stats;
    }

    void GenerateReport(int crashCount) {
        std::wcout << L"========================================\n";
        std::wcout << L"           稳定性测试报告\n";
        std::wcout << L"========================================\n\n";

        // 提取各指标数据
        std::vector<double> cpuValues, memValues, privateValues, gdiValues, userValues, threadValues, pageFaultValues;
        int runningCount = 0;

        for (const auto& s : m_samples) {
            if (s.isRunning) {
                runningCount++;
                cpuValues.push_back(s.cpuUsage);
                memValues.push_back(static_cast<double>(s.workingSetMB));
                privateValues.push_back(static_cast<double>(s.privateBytesMB));
                gdiValues.push_back(static_cast<double>(s.gdiHandles));
                userValues.push_back(static_cast<double>(s.userHandles));
                threadValues.push_back(static_cast<double>(s.threadCount));
                pageFaultValues.push_back(static_cast<double>(s.pageFaults));
            }
        }

        std::wcout << L"【基本信息】\n";
        std::wcout << L"  总采样数: " << m_samples.size() << L"\n";
        std::wcout << L"  有效采样: " << runningCount << L"\n";
        std::wcout << L"  进程崩溃/停止次数: " << crashCount << L"\n";
        std::wcout << L"  运行时间: " << m_durationSec << L" 秒\n\n";

        if (runningCount == 0) {
            std::wcout << L"  没有收集到有效数据！\n";
            return;
        }

        // 打印各指标统计
        auto printStats = [](const std::wstring& name, const Statistics& stats, const std::wstring& unit, bool showTrend = true) {
            std::wcout << L"【" << name << L"】\n";
            std::wcout << std::fixed << std::setprecision(2);
            std::wcout << L"  最小值: " << stats.min << L" " << unit << L"\n";
            std::wcout << L"  最大值: " << stats.max << L" " << unit << L"\n";
            std::wcout << L"  平均值: " << stats.avg << L" " << unit << L"\n";
            std::wcout << L"  标准差: " << stats.stdDev << L" " << unit << L"\n";
            if (showTrend) {
                std::wcout << L"  趋势: " << (stats.trend > 0.001 ? L"↑ 上升" : (stats.trend < -0.001 ? L"↓ 下降" : L"→ 稳定"));
                std::wcout << L" (" << std::showpos << stats.trend << std::noshowpos << L"/采样)\n";
            }
            std::wcout << L"\n";
        };

        printStats(L"CPU 使用率", CalculateStats(cpuValues), L"%");
        printStats(L"工作集内存", CalculateStats(memValues), L"MB");
        printStats(L"私有内存", CalculateStats(privateValues), L"MB");
        printStats(L"GDI 句柄", CalculateStats(gdiValues), L"个");
        printStats(L"USER 句柄", CalculateStats(userValues), L"个");
        printStats(L"线程数", CalculateStats(threadValues), L"个", false);

        // 页面错误增长分析
        if (pageFaultValues.size() >= 2) {
            double pageFaultGrowth = pageFaultValues.back() - pageFaultValues.front();
            double growthPerSec = pageFaultGrowth / m_durationSec;
            std::wcout << L"【页面错误】\n";
            std::wcout << L"  初始值: " << static_cast<DWORD>(pageFaultValues.front()) << L"\n";
            std::wcout << L"  最终值: " << static_cast<DWORD>(pageFaultValues.back()) << L"\n";
            std::wcout << L"  增长量: " << static_cast<DWORD>(pageFaultGrowth) << L"\n";
            std::wcout << L"  增长率: " << growthPerSec << L"/秒\n\n";
        }

        // 稳定性评估
        std::wcout << L"========================================\n";
        std::wcout << L"           稳定性评估结论\n";
        std::wcout << L"========================================\n\n";

        int score = 100;
        std::vector<std::wstring> issues;

        // 评分规则
        if (crashCount > 0) {
            score -= 50;
            issues.push_back(L"进程发生崩溃或停止");
        }

        auto memStats = CalculateStats(memValues);
        if (memStats.trend > 0.1) {
            score -= 20;
            issues.push_back(L"内存持续上升，可能存在内存泄漏");
        }

        auto gdiStats = CalculateStats(gdiValues);
        if (gdiStats.trend > 0.01) {
            score -= 15;
            issues.push_back(L"GDI 句柄持续增加，可能存在 GDI 泄漏");
        }

        auto cpuStats = CalculateStats(cpuValues);
        if (cpuStats.avg > 10) {
            score -= 10;
            issues.push_back(L"CPU 使用率偏高");
        }
        if (cpuStats.stdDev > 5) {
            score -= 5;
            issues.push_back(L"CPU 使用率波动较大");
        }

        if (memStats.max > 100) {
            score -= 5;
            issues.push_back(L"内存占用较高 (>100MB)");
        }

        // 输出评分
        std::wcout << L"  稳定性评分: " << score << L"/100\n\n";
        
        if (score >= 90) {
            std::wcout << L"  评级: ★★★★★ 优秀\n";
        } else if (score >= 75) {
            std::wcout << L"  评级: ★★★★☆ 良好\n";
        } else if (score >= 60) {
            std::wcout << L"  评级: ★★★☆☆ 一般\n";
        } else if (score >= 40) {
            std::wcout << L"  评级: ★★☆☆☆ 较差\n";
        } else {
            std::wcout << L"  评级: ★☆☆☆☆ 差\n";
        }

        if (!issues.empty()) {
            std::wcout << L"\n  发现的问题:\n";
            for (const auto& issue : issues) {
                std::wcout << L"    - " << issue << L"\n";
            }
        } else {
            std::wcout << L"\n  未发现明显问题，程序运行稳定。\n";
        }

        std::wcout << L"\n========================================\n";
    }

    void SaveToCSV() {
        std::string filename = "stability_report_" + 
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".csv";
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::wcerr << L"无法创建 CSV 文件\n";
            return;
        }

        // UTF-8 BOM
        file << "\xEF\xBB\xBF";
        
        // 表头
        file << "Timestamp(s),CPU(%),WorkingSet(MB),PrivateBytes(MB),GDI,USER,Threads,PageFaults,Status\n";
        
        // 数据
        for (const auto& s : m_samples) {
            file << std::fixed << std::setprecision(2)
                 << s.timestamp << ","
                 << s.cpuUsage << ","
                 << s.workingSetMB << ","
                 << s.privateBytesMB << ","
                 << s.gdiHandles << ","
                 << s.userHandles << ","
                 << s.threadCount << ","
                 << s.pageFaults << ","
                 << (s.isRunning ? "Running" : "Stopped")
                 << "\n";
        }

        file.close();
        std::wcout << L"\n数据已保存到: " << std::wstring(filename.begin(), filename.end()) << L"\n";
    }

private:
    std::wstring m_processName;
    int m_durationSec;
    int m_intervalMs;
    DWORD m_processId = 0;
    HANDLE m_hProcess = nullptr;
    DWORD m_numProcessors = 1;
    
    UINT64 m_lastKernelTime = 0;
    UINT64 m_lastUserTime = 0;
    std::chrono::high_resolution_clock::time_point m_lastCheckTime;
    
    std::vector<SampleData> m_samples;
};

int main(int argc, char* argv[]) {
    // 设置控制台输出为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    
    // 解析参数
    int duration = 60;  // 默认 60 秒
    if (argc >= 2) {
        duration = std::atoi(argv[1]);
        if (duration <= 0) duration = 60;
    }

    int interval = 500;  // 默认 500ms
    if (argc >= 3) {
        interval = std::atoi(argv[2]);
        if (interval < 100) interval = 100;
    }

    StabilityMonitor monitor(L"dx_overlay.exe", duration, interval);
    
    if (!monitor.Run()) {
        std::wcerr << L"\n监控失败，请确保 dx_overlay.exe 正在运行。\n";
        return 1;
    }

    std::wcout << L"\n按任意键退出...\n";
    std::cin.get();
    return 0;
}
