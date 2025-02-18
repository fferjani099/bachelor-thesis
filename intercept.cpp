#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include <cuda.h>
#include <libsmctrl.h>

/**
 * This library intercepts CUDA Driver calls (like cuLaunchKernel).
 * We create new CUDA streams for each Linux thread, and apply TPC-masking
 * so that each thread's GPU kernels only run on a distinct subset of TPCs.
 *
 * Usage:
 *   - Build as a shared library (e.g. `libgpu_intercept.so`).
 *   - Preload it in the environment:
 *       LD_PRELOAD=/path/to/libgpu_intercept.so ...
 *   - Each distinct TID that invokes a kernel will get a unique TPC subset.
 */

// ---------------------------------------------------------------------
// Global / static variables
// ---------------------------------------------------------------------

static bool g_initialized = false;
static std::mutex g_init_mutex;

// Original function pointer
static CUresult (*orig_cuLaunchKernel)(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra) = nullptr;

// Per‐thread stream map
static std::map<pid_t, CUstream> g_thread_stream_map;
static std::mutex g_map_mutex;

// Number of TPCs on the device
static uint32_t g_num_tpcs = 2;

// Number of partitions (groups)
static uint32_t g_num_groups = 2; // adjust as needed

// Atomic counter for group assignment
static std::atomic<uint32_t> g_next_group_index{0};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

static void load_original_symbols()
{
    if (!orig_cuLaunchKernel) {
        orig_cuLaunchKernel = reinterpret_cast<CUresult(*)(CUfunction, 
            unsigned int, unsigned int, unsigned int,
            unsigned int, unsigned int, unsigned int, 
            unsigned int, CUstream, void**, void**)>(
            dlsym(RTLD_NEXT, "cuLaunchKernel")
        );
        if (!orig_cuLaunchKernel) {
            fprintf(stderr,
                "[gpu_intercept_driver] ERROR: Failed to find original cuLaunchKernel.\n");
            _exit(1);
        }
    }
}

// Build a bitmask [start_tpc..end_tpc], then invert for the “disabled bits”.
static uint64_t build_tpc_mask(uint32_t start_tpc, uint32_t end_tpc)
{
    if (start_tpc > end_tpc || start_tpc >= 64)
        return 0ULL;
    if (end_tpc >= 64)
        end_tpc = 63;
    uint64_t mask = 0ULL;
    for (uint32_t t = start_tpc; t <= end_tpc; t++) {
        mask |= (1ULL << t);
    }
    return ~mask; // libsmctrl uses "1 bits => disabled"
}

static uint64_t compute_mask_for_tid(pid_t tid)
{
    // Round‐robin assignment to g_num_groups
    uint32_t group_index =
        g_next_group_index.fetch_add(1, std::memory_order_relaxed) % g_num_groups;

    if (g_num_tpcs == 0)
        return 0ULL;

    uint32_t base_count = g_num_tpcs / g_num_groups;
    uint32_t remainder  = g_num_tpcs % g_num_groups;

    uint32_t start = 0;
    uint32_t count = 0;

    for (uint32_t g = 0; g < g_num_groups; g++) {
        uint32_t extra = (g < remainder) ? 1U : 0U;
        uint32_t group_size = base_count + extra;
        if (g == group_index) {
            count = group_size;
            break;
        }
        start += group_size;
    }

    if (!count) {
        return 0ULL;
    }
    uint32_t end = start + count - 1;
    if (end >= g_num_tpcs)
        end = g_num_tpcs - 1;

    uint64_t mask = build_tpc_mask(start, end);
    fprintf(stdout, "[gpu_intercept_driver] TID %d => group=%u => TPC range [%u..%u], mask=0x%lx\n",
            tid, group_index, start, end, mask);
    return mask;
}

static CUstream get_or_create_thread_stream()
{
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));

    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        auto it = g_thread_stream_map.find(tid);
        if (it != g_thread_stream_map.end()) {
            return it->second;
        }
    }

    CUstream new_stream = nullptr;
    CUresult cuerr = cuStreamCreate(&new_stream, CU_STREAM_DEFAULT);
    if (cuerr != CUDA_SUCCESS) {
        fprintf(stderr,
            "[gpu_intercept_driver] ERROR: cuStreamCreate failed for TID=%d, code=%d\n",
            tid, (int)cuerr);
        _exit(1);
    }

    uint64_t stream_mask = compute_mask_for_tid(tid);
    libsmctrl_set_stream_mask((void*)new_stream, stream_mask);

    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        g_thread_stream_map[tid] = new_stream;
    }

    fprintf(stdout,
        "[gpu_intercept_driver] Created new CUstream=%p for TID=%d with mask=0x%lx\n",
        (void*)new_stream, tid, stream_mask);
    return new_stream;
}

static void do_init()
{
    std::lock_guard<std::mutex> lk(g_init_mutex);
    if (g_initialized) return;

    load_original_symbols();

    CUresult cuerr = cuInit(0);
    if (cuerr != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu_intercept_driver] cuInit(0) failed with code %d.\n", (int)cuerr);
        _exit(1);
    }

    int dev = 0;
    CUdevice cuDev;
    if ((cuerr = cuDeviceGet(&cuDev, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu_intercept_driver] cuDeviceGet(%d) failed.\n", dev);
        _exit(1);
    }
    CUcontext ctx;
    if ((cuerr = cuCtxCreate(&ctx, 0, cuDev)) != CUDA_SUCCESS) {
        fprintf(stderr, "[gpu_intercept_driver] cuCtxCreate failed for dev=%d.\n", dev);
        _exit(1);
    }

    // libsmctrl_get_tpc_info(&g_num_tpcs, dev);  // if you want the real TPC count.
    fprintf(stdout, "[gpu_intercept_driver] GPU device %d => %u TPC(s)\n", dev, g_num_tpcs);

    // Optionally set a global mask (zero => no TPCs are globally disabled).
    uint64_t global_mask = 0ULL;
    libsmctrl_set_global_mask(global_mask);

    fprintf(stdout, "[gpu_intercept_driver] Initialization complete.\n");
    g_initialized = true;
}

// ---------------------------------------------------------------------
// Hooked driver functions
// ---------------------------------------------------------------------

extern "C"
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra)
{
    do_init();

    CUstream my_stream = get_or_create_thread_stream();
    // Overwrite the user’s stream with ours:
    return orig_cuLaunchKernel(f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        my_stream,
        kernelParams,
        extra);
}

// ---------------------------------------------------------------------
// Library constructor / destructor
// ---------------------------------------------------------------------

__attribute__((constructor))
static void lib_constructor()
{
    // Called on library load, but we do lazy init in do_init().
}

__attribute__((destructor))
static void lib_destructor()
{
    // Destroy all per‐thread streams
    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        for (auto& kv : g_thread_stream_map) {
            cuStreamDestroy(kv.second);
        }
        g_thread_stream_map.clear();
    }
    fprintf(stdout, "[gpu_intercept_driver] Interceptor library shutdown.\n");
}