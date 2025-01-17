#define _GNU_SOURCE
#include <cuda_runtime_api.h>
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
#include <sstream>
#include <thread>

#include <libsmctrl.h>

/**
 * This library intercepts CUDA runtime calls (like cudaLaunchKernel).
 * We create new CUDA streams for each Linux thread, and apply TPC-masking
 * so that each thread's GPU kernels only run on a distinct subset of TPCs.
 *
 * Usage:
 *   - Build as a shared library (e.g. `intercept.so`).
 *   - Preload it in the environment:
 *       LD_PRELOAD=/path/to/intercept.so ...
 *   - It intercepts all calls to cudaLaunchKernel in the application.
 *   - Each distinct TID that invokes a kernel will get a unique TPC subset.
 */

// ---------------------------------------------------------------------
// Global / static variables
// ---------------------------------------------------------------------

static bool g_initialized = false;
static std::mutex g_init_mutex;

// Points to the true/original cudaLaunchKernel
static cudaError_t (*original_cudaLaunchKernel)(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream) = nullptr;

// We map each TID -> dedicated CUDA stream
static std::map<pid_t, cudaStream_t> g_thread_stream_map;
static std::mutex g_map_mutex;

// Number of TPCs on the device
static uint32_t g_num_tpcs = 0;

// Number of distinct partitions we want to form. For example: 2 or 4.
static uint32_t g_num_groups = 2;  // adjust as needed

// Atomic counter to assign group indices to newly seen threads
static std::atomic<uint32_t> g_next_group_index{0};

// ---------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------

static void initialize_if_needed() {
    std::lock_guard<std::mutex> lk(g_init_mutex);
    if (g_initialized) {
        return;
    }

    // Resolve the original symbol for cudaLaunchKernel
    original_cudaLaunchKernel = reinterpret_cast<cudaError_t(*)(const void*, dim3, dim3, void**, size_t, cudaStream_t)>(
        dlsym(RTLD_NEXT, "cudaLaunchKernel")
    );

    if (!original_cudaLaunchKernel) {
        fprintf(stderr, "[gpu_intercept] ERROR: Unable to find original cudaLaunchKernel.\n");
        _exit(1);
    }

    // Initialize the device and libsmctrl
    int dev = 0;
    cudaError_t cerr = cudaSetDevice(dev);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[gpu_intercept] ERROR: cudaSetDevice(dev=%d) failed.\n", dev);
        _exit(1);
    }

    // Query how many TPCs exist on this device
    libsmctrl_get_tpc_info(&g_num_tpcs, dev);
    //g_num_tpcs = 60; 
    fprintf(stdout, "[gpu_intercept] Device %d has %u TPC(s)\n", dev, g_num_tpcs);

    // set a global mask
    // we disable none by default => global_mask = 0, i.e. no TPCs are disabled
    uint64_t global_mask = 0x0ULL; 
    libsmctrl_set_global_mask(global_mask);

    // Mark initialization done
    g_initialized = true;
    fprintf(stdout, "[gpu_intercept] Initialization complete. \n");
}

// ---------------------------------------------------------------------
// Building a bitmask for the TPC subset
//  - If start_tpc <= end_tpc < g_num_tpcs, set bits [start_tpc..end_tpc]
// ---------------------------------------------------------------------
static uint64_t build_tpc_mask(uint32_t start_tpc, uint32_t end_tpc) {
    uint64_t mask = 0ULL;
    if (start_tpc > end_tpc || start_tpc >= 64) {
        // no valid range or beyond 64 bits
        return mask;
    }
    // end_tpc must be <= 63 for 64-bit mask. We'll clamp if needed
    if (end_tpc >= 64) {
        end_tpc = 63;
    }
    for (uint32_t t = start_tpc; t <= end_tpc; t++) {
        mask |= (1ULL << t);
    }
    return ~mask;
}

// ---------------------------------------------------------------------
// Assign a group index to a TID and build an appropriate mask
// ---------------------------------------------------------------------
static uint64_t compute_mask_for_tid(pid_t tid) {
    // We'll do round-robin assignment:
    //    group_index = (g_next_group_index++) % g_num_groups
    // Then we compute how many TPCs belong in each group.

    // The simplest approach is to do "uniform" distribution of TPCs among groups.
    // E.g. if we have 32 TPCs and 2 groups => each group gets 16 TPCs
    // If not divisible, the last group can get remainder.

    uint32_t group_index = g_next_group_index.fetch_add(1, std::memory_order_relaxed) % g_num_groups;
    if (g_num_tpcs == 0) {
        return 0ULL;  // no TPCs? no bits
    }

    // integer division: each group has at least base_count TPCs
    uint32_t base_count = g_num_tpcs / g_num_groups;
    // remainder TPCs distributed across the first (g_num_tpcs % g_num_groups) groups
    uint32_t remainder = g_num_tpcs % g_num_groups;

    // We'll compute how many TPCs come before this group in the assignment
    // and how many TPCs are assigned to this group
    uint32_t start = 0, count = 0;

    for (uint32_t g = 0; g < g_num_groups; g++) {
        uint32_t extra = (g < remainder) ? 1U : 0U; 
        uint32_t group_size = base_count + extra;
        if (g == group_index) {
            // found our group
            count = group_size;
            break;
        }
        // else move start by group_size
        start += group_size;
    }

    if (count == 0) {
        // means maybe we had 0 TPC or something
        return 0ULL;
    }

    uint32_t end = start + count - 1;
    if (end >= g_num_tpcs) {
        end = g_num_tpcs - 1;
    }

    // build bitmask
    uint64_t mask = build_tpc_mask(start, end);

    fprintf(stdout,
            "[gpu_intercept] TID=%d assigned group_index=%u => TPC range [%u..%u], mask=0x%lx\n",
            tid, group_index, start, end, mask);
    return mask;
}

// ---------------------------------------------------------------------
// Get or create a dedicated CUDA stream for the calling TID
// ---------------------------------------------------------------------
static cudaStream_t get_or_create_thread_stream() {
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));

    // First check if we already have a stream
    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        auto it = g_thread_stream_map.find(tid);
        if (it != g_thread_stream_map.end()) {
            return it->second;
        }
    }

    // If not found, create a new stream
    cudaStream_t new_stream;
    cudaError_t err = cudaStreamCreate(&new_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_intercept] ERROR: cudaStreamCreate() failed for TID=%d\n", tid);
        _exit(1);
    }

    // Compute a TPC mask for this TID
    uint64_t stream_mask = compute_mask_for_tid(tid);

    // Apply this mask to the new stream
    libsmctrl_set_stream_mask(new_stream, stream_mask);

    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        g_thread_stream_map[tid] = new_stream;
    }

    fprintf(stdout,
            "[gpu_intercept] Created new stream=%p for TID=%d with mask=0x%lx\n",
            (void*)new_stream, tid, stream_mask);

    return new_stream;
}

// ---------------------------------------------------------------------
// Intercepted cudaLaunchKernel
// We override the user-specified stream with our own partitioned stream
// ---------------------------------------------------------------------
extern "C"
cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream)
{
    // Make sure we're set up
    initialize_if_needed();

    // Retrieve or create the thread’s dedicated stream
    cudaStream_t my_stream = get_or_create_thread_stream();

    // Force the kernel to launch on our partitioned stream
    return original_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, my_stream);
}

// ---------------------------------------------------------------------
// Library constructor and destructor
// ---------------------------------------------------------------------
__attribute__((constructor))
static void lib_init() {
    // Called when the shared library is loaded.
    // We do lazy initialization in the interceptor, so nothing special here.
}

__attribute__((destructor))
static void lib_fini() {
    // Cleanup on library unload. Destroy all streams, etc.
    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        for (auto &kv : g_thread_stream_map) {
            cudaStreamDestroy(kv.second);
        }
        g_thread_stream_map.clear();
    }
    fprintf(stdout, "[gpu_intercept] Library shutdown complete.\n");
}