#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <thread>
#include <sstream>

#include <libsmctrl.h>

// This library intercepts CUDA runtime calls (like cudaLaunchKernel).
// We'll create new CUDA streams per thread and assign masks to them.


static bool g_initialized = false;
static std::mutex g_init_mutex;
static std::mutex g_map_mutex;

// Map thread_id to cudaStream_t
static std::map<pid_t, cudaStream_t> g_thread_stream_map;

// The original cudaLaunchKernel function pointer
static cudaError_t (*original_cudaLaunchKernel)(
  const void *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem,
  cudaStream_t stream) = nullptr;

// Device info
static uint32_t g_num_tpcs = 0;

// Global and per-stream mask (for demonstration, we can pick a simple mask)
static uint64_t g_global_mask = 0; 
// For example, if we have 8 TPCs and we want to enable all except one:
// g_global_mask = 0x02 would disable TPC #1 (assuming bit 0 = TPC0).
 

static void initialize_if_needed() {
    std::lock_guard<std::mutex> lk(g_init_mutex);
    if (g_initialized) return;

    original_cudaLaunchKernel = (cudaError_t(*)(const void*, dim3, dim3, void**, size_t, cudaStream_t))
        dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if (!original_cudaLaunchKernel) {
        fprintf(stderr, "[gpu_intercept] Error: Unable to find original cudaLaunchKernel.\n");
        _exit(1);
    }

    // Initialize libsmctrl: Obtain device info
    int dev = 0; // Assuming default device 0
    cudaSetDevice(dev);

    libsmctrl_get_tpc_info(&g_num_tpcs, dev);

    // we recommend disabling most TPCs by default. 
    // CUDA may implicitly launch internal kernels to support some API calls and, if no default mask is set, those calls may interfere with the partitions
    // Allow work to only use TPC 1 by default
    libsmctrl_set_global_mask(~0x1ull);
    // It is possible to disable all TPCs by default (with a mask of ~0ull)
    // but we recommend against this, as it causes kernels launched with the default TPC mask to hang indefinitely (including CUDA-internal ones)

    g_initialized = true;
    fprintf(stdout, "[gpu_intercept] Initialization complete. Total TPCs: %u\n", g_num_tpcs);
}

static cudaStream_t get_or_create_thread_stream() {
    pid_t tid = (pid_t) syscall(SYS_gettid); // using the Linux TID

    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        auto it = g_thread_stream_map.find(tid);
        if (it != g_thread_stream_map.end()) {
            return it->second;
        }
    }

    cudaStream_t new_stream;
    cudaError_t err = cudaStreamCreate(&new_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_intercept] Failed to create CUDA stream for thread %d\n", tid);
        _exit(1);
    }

    // Let's pick a group size of 10 TPCs per Task.
    const uint32_t group_size = 10;
    if (g_num_tpcs < group_size) {
        fprintf(stderr, "[gpu_intercept] Warning: fewer than 10 TPCs available, using all.\n");
        // Just use all TPCs if we don't have enough.
        // mask: bits set for all TPCs
        uint64_t stream_mask = (g_num_tpcs >= 64) ? (uint64_t)-1 : ((1ULL << g_num_tpcs) - 1);
        libsmctrl_set_stream_mask(new_stream, stream_mask);
        std::lock_guard<std::mutex> lk(g_map_mutex);
        g_thread_stream_map[tid] = new_stream;
        fprintf(stdout, "[gpu_intercept] Created stream %p for thread %d with mask 0x%lx\n", 
                (void*)new_stream, tid, stream_mask);
        return new_stream;
    }

    uint32_t num_groups = g_num_tpcs / group_size;
    if (num_groups == 0) num_groups = 1; // Fallback if less than 10 TPCs.

    uint32_t group_index = (uint32_t)(tid % num_groups);

    // Calculate the mask for this group
    // Example: if group_index = 1 and group_size = 10, 
    // TPCs [10..19] are enabled.
    uint64_t stream_mask = ((1ULL << group_size) - 1) << (group_index * group_size);
    // Make sure we don't shift beyond the number of TPCs if not perfectly divisible
    stream_mask &= (g_num_tpcs >= 64) ? (uint64_t)-1 : ((1ULL << g_num_tpcs) - 1);

    libsmctrl_set_stream_mask(new_stream, stream_mask);

    {
        std::lock_guard<std::mutex> lk(g_map_mutex);
        g_thread_stream_map[tid] = new_stream;
    }

    fprintf(stdout, "[gpu_intercept] Created stream %p for thread %d with mask 0x%lx (TPC group %u)\n", 
            (void*)new_stream, tid, stream_mask, group_index);

    return new_stream;
}


extern "C" cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream)
{
    initialize_if_needed();
    stream = get_or_create_thread_stream();

    return original_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}


// ============================================================================
// Library constructor and destructor
// ============================================================================
__attribute__((constructor))
static void lib_init() {
    // We do lazy init in initialize_if_needed(), so nothing here, but we could.
    // This is called when the shared library is loaded.
}

__attribute__((destructor))
static void lib_fini() {
    // Cleanup if needed.
    // Destroy streams, etc.
    for (auto &kv : g_thread_stream_map) {
        cudaStreamDestroy(kv.second);
    }
    g_thread_stream_map.clear();
    fprintf(stdout, "[gpu_intercept] Library shutdown.\n");
}
