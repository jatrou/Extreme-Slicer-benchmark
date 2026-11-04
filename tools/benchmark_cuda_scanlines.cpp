// Standalone benchmark for the production CUDA vertical-intersection dispatch.
// Build instructions and measurement definitions: benchmark_cuda_scanlines.md.
#include "../src/libslic3r/Gpu/CudaSlicer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace Slic3r::Gpu;
using Clock = std::chrono::steady_clock;

// Prevent link-time optimization from folding repeated CPU trials away.
#if defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE __attribute__((noinline))
#endif

BENCH_NOINLINE void cpu_intersections(
    const std::vector<VulkanVerticalIntersectionRequest>& requests,
    std::vector<VulkanVerticalIntersection>& results)
{
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& r = requests[i];
        const int64_t ax = r.segment.a.x;
        const int64_t bx = r.segment.b.x;
        VulkanVerticalIntersection value { 0, 1, r.stable_id, false };
        if (ax != bx && r.scan_x > std::min(ax, bx) && r.scan_x < std::max(ax, bx)) {
            const bool forward = bx > ax;
            const int64_t t = forward ? r.scan_x - ax : ax - r.scan_x;
            const int64_t denominator = forward ? bx - ax : ax - bx;
            value.numerator = t * (r.segment.b.y - r.segment.a.y) +
                              r.segment.a.y * denominator;
            value.denominator = denominator;
            value.valid = true;
        }
        results[i] = value;
    }
}

std::vector<VulkanVerticalIntersectionRequest> make_requests(size_t count, bool include_misses = false)
{
    std::vector<VulkanVerticalIntersectionRequest> requests;
    requests.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        // One fixed-point unit is 1 nm. Cover a 300 mm build plate while
        // keeping the numerator products well within signed 64-bit range.
        const int64_t left = int64_t((i * 79'123'457ULL + 61'234'567ULL) % 180'000'001ULL);
        const int64_t width = 2'001 + int64_t((i * 43'791'923ULL + 28'245'912ULL) % 119'997'999ULL);
        const int64_t right = left + width;
        const int64_t ay = int64_t((i * 137'221'119ULL + 73'001'937ULL) % 300'000'001ULL);
        const int64_t by = int64_t((i * 177'409'033ULL + 219'000'617ULL) % 300'000'001ULL);
        int64_t scan_x = left + 1 + int64_t((i * 29'011'337ULL + 617ULL) % uint64_t(width - 1));
        if (include_misses && i % 19 == 1)
            scan_x = left; // endpoint is excluded
        else if (include_misses && i % 19 == 2)
            scan_x = right + 1; // outside the segment
        const bool reverse = (i & 1) != 0;
        const int64_t ax = include_misses && i % 19 == 0 ? left : (reverse ? right : left);
        const int64_t bx = include_misses && i % 19 == 0 ? left : (reverse ? left : right);
        requests.push_back({ { { ax, ay }, { bx, by } }, scan_x,
                             uint64_t(i) * 13 + 5 });
    }
    return requests;
}

uint64_t checksum(const std::vector<VulkanVerticalIntersection>& values)
{
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& value : values) {
        hash = (hash ^ uint64_t(value.numerator)) * 1099511628211ULL;
        hash = (hash ^ uint64_t(value.denominator)) * 1099511628211ULL;
        hash = (hash ^ value.stable_id) * 1099511628211ULL;
        hash = (hash ^ uint64_t(value.valid)) * 1099511628211ULL;
    }
    return hash;
}

void validate(const std::vector<VulkanVerticalIntersection>& expected,
              const VulkanVerticalIntersectionBatch& actual, size_t count)
{
    if (!actual.dispatched)
        throw std::runtime_error("CUDA dispatch failed at n=" + std::to_string(count) + ": " + actual.diagnostic);
    if (actual.intersections.size() != expected.size())
        throw std::runtime_error("CUDA returned the wrong result count at n=" + std::to_string(count));
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& a = expected[i];
        const auto& b = actual.intersections[i];
        if (a.numerator != b.numerator || a.denominator != b.denominator ||
            a.stable_id != b.stable_id || a.valid != b.valid)
            throw std::runtime_error("CPU/CUDA mismatch at n=" + std::to_string(count) +
                                     ", index=" + std::to_string(i));
    }
}

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() & 1 ? samples[middle] :
           (samples[middle - 1] + samples[middle]) / 2.0;
}

struct Row {
    size_t count;
    int repetitions;
    size_t cpu_passes;
    double cpu_ms;
    double cuda_full_ms;
    double cuda_launch_sync_ms;
    uint64_t digest;
};

Row benchmark(size_t count, int repetitions)
{
    const auto requests = make_requests(count);
    std::vector<VulkanVerticalIntersection> expected(count);
    cpu_intersections(requests, expected);
    const uint64_t digest = checksum(expected);

    // Target several million CPU operations per sample, with a minimum of one
    // pass. Report per-pass latency, rather than the aggregate loop duration.
    const size_t cpu_passes = std::max<size_t>(1, 4'000'000 / count);
    std::vector<double> cpu_times, cuda_times, launch_sync_times;
    cpu_times.reserve(repetitions);
    cuda_times.reserve(repetitions);
    launch_sync_times.reserve(repetitions);
    std::vector<VulkanVerticalIntersection> cpu_output(count);
    for (int rep = 0; rep < repetitions; ++rep) {
        const auto cpu_start = Clock::now();
        for (size_t pass = 0; pass < cpu_passes; ++pass)
            cpu_intersections(requests, cpu_output);
        cpu_times.push_back(elapsed_ms(cpu_start) / double(cpu_passes));
        if (checksum(cpu_output) != digest)
            throw std::runtime_error("CPU checksum changed at n=" + std::to_string(count));

        // Includes production packing, allocations, both copies, launch,
        // synchronization, freeing and output materialization.
        const auto cuda_start = Clock::now();
        const auto batch = CudaSlicerBackend::dispatch_vertical_intersections(requests);
        cuda_times.push_back(elapsed_ms(cuda_start));
        validate(expected, batch, count);
        // Production code times kernel launch + cudaDeviceSynchronize with a
        // host clock. It is not an isolated GPU-event kernel execution time.
        launch_sync_times.push_back(batch.gpu_elapsed_ms);
    }
    return { count, repetitions, cpu_passes, median(cpu_times), median(cuda_times),
             median(launch_sync_times), digest };
}

void print_row_csv(const Row& row)
{
    std::cout << "warm," << row.count << ',' << row.repetitions << ',' << row.cpu_passes << ','
              << row.cpu_ms << ',' << row.cuda_full_ms << ',' << row.cuda_launch_sync_ms << ','
              << (row.cuda_full_ms / row.cpu_ms) << ',' << row.digest << "\n";
}
} // namespace

int main(int argc, char** argv)
{
    try {
        int repetitions = 11;
        bool json = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--repetitions" && i + 1 < argc) {
                repetitions = std::stoi(argv[++i]);
                if (repetitions < 3 || repetitions > 1000)
                    throw std::runtime_error("--repetitions must be 3..1000");
            } else if (arg == "--format" && i + 1 < argc) {
                const std::string format = argv[++i];
                if (format != "csv" && format != "json")
                    throw std::runtime_error("--format must be csv or json");
                json = format == "json";
            } else {
                throw std::runtime_error("usage: benchmark_cuda_scanlines [--repetitions N] [--format csv|json]");
            }
        }
        std::cout << std::fixed << std::setprecision(9);

        // First CUDA API use happens inside this dispatch. Keep this distinct
        // from warmed medians; do not pre-query or pre-initialize the device.
        const auto cold_requests = make_requests(64);
        std::vector<VulkanVerticalIntersection> cold_expected(64);
        cpu_intersections(cold_requests, cold_expected);
        const auto cold_start = Clock::now();
        const auto cold_batch = CudaSlicerBackend::dispatch_vertical_intersections(cold_requests);
        const double cold_ms = elapsed_ms(cold_start);
        validate(cold_expected, cold_batch, 64);

        const auto capabilities = CudaSlicerBackend::query_capabilities();
        if (!capabilities.runtime_available)
            throw std::runtime_error(capabilities.diagnostic);

        // Keep production-like timed batches strict-interior. Check the
        // excluded cases separately after recording the cold dispatch.
        auto edge_requests = make_requests(2048, true);
        const size_t base_count = edge_requests.size();
        for (int64_t offset : { -300'000'000LL, 300'000'000LL }) {
            for (size_t i = 0; i < base_count; ++i) {
                auto translated = edge_requests[i];
                translated.segment.a.x += offset;
                translated.segment.a.y += offset;
                translated.segment.b.x += offset;
                translated.segment.b.y += offset;
                translated.scan_x += offset;
                translated.stable_id = uint64_t(edge_requests.size()) * 13 + 5;
                edge_requests.push_back(translated);
            }
        }
        std::vector<VulkanVerticalIntersection> edge_expected(edge_requests.size());
        cpu_intersections(edge_requests, edge_expected);
        validate(edge_expected,
                 CudaSlicerBackend::dispatch_vertical_intersections(edge_requests),
                 edge_requests.size());

        const size_t sizes[] = { 64, 512, 4096, 16384, 65536, 262144 };
        std::vector<Row> rows;
        rows.reserve(std::size(sizes));
        for (size_t count : sizes)
            rows.push_back(benchmark(count, repetitions));

        if (json) {
            std::cout << "{\"schema\":1,\"device\":\"" << capabilities.device_name
                      << "\",\"cold_n\":64,\"cold_full_ms\":" << cold_ms
                      << ",\"cold_launch_sync_ms\":" << cold_batch.gpu_elapsed_ms
                      << ",\"rows\":[";
            for (size_t i = 0; i < rows.size(); ++i) {
                const auto& r = rows[i];
                if (i) std::cout << ',';
                std::cout << "{\"n\":" << r.count << ",\"repetitions\":" << r.repetitions
                          << ",\"cpu_passes\":" << r.cpu_passes << ",\"cpu_ms\":" << r.cpu_ms
                          << ",\"cuda_full_ms\":" << r.cuda_full_ms
                          << ",\"cuda_launch_sync_ms\":" << r.cuda_launch_sync_ms
                          << ",\"cuda_over_cpu\":" << (r.cuda_full_ms / r.cpu_ms)
                          << ",\"checksum\":" << r.digest << '}';
            }
            std::cout << "]}\n";
        } else {
            std::cout << "phase,n,repetitions,cpu_passes,cpu_ms,cuda_full_ms,cuda_launch_sync_ms,cuda_over_cpu,checksum\n";
            std::cout << "cold,64,1,0,0," << cold_ms << ',' << cold_batch.gpu_elapsed_ms
                      << ",0," << checksum(cold_expected) << "\n";
            for (const auto& row : rows)
                print_row_csv(row);
        }
        CudaSlicerBackend::release_resources();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cuda_scanlines: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
