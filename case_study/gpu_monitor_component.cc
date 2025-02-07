#include "gpu_monitor_component.h"

bool GpuMonitorComponent::Init() {
  nvmlReturn_t ret = nvmlInit_v2();
  if (ret != NVML_SUCCESS) {
    AERROR << "[GpuMonitorComponent] NVML init error: " << nvmlErrorString(ret);
    return false;
  }
  nvml_initialized_ = true;

  // Grab device 0
  ret = nvmlDeviceGetHandleByIndex(0, &device_);
  if (ret != NVML_SUCCESS) {
    AERROR << "[GpuMonitorComponent] Could not get device handle by index 0: "
           << nvmlErrorString(ret);
    return false;
  }

  AINFO << "[GpuMonitorComponent] NVML init successful; device 0 handle acquired.";
  return true;
}

bool GpuMonitorComponent::Proc() {
  if (!nvml_initialized_) {
    AERROR << "[GpuMonitorComponent] NVML not initialized. Exiting.";
    return false;
  }

  nvmlUtilization_t utilization;
  nvmlReturn_t ret = nvmlDeviceGetUtilizationRates(device_, &utilization);
  if (ret != NVML_SUCCESS) {
    AERROR << "[GpuMonitorComponent] nvmlDeviceGetUtilizationRates error: "
           << nvmlErrorString(ret);
    return false;
  }

  // usage is a percentage
  float gpu_usage = static_cast<float>(utilization.gpu);

  // Log usage for this iteration
  AINFO << "[GpuMonitor] GPU usage: " << gpu_usage << "%, ";

  // Accumulate totals
  iteration_count_++;
  total_gpu_usage_ += gpu_usage;

  // Every 100 iterations, log average usage
  if (iteration_count_ == 100) {
    float avg_gpu = total_gpu_usage_ / 100.0f;

    AINFO << "=========================================================";
    AINFO << "[GpuMonitor] **Average** over last 100 iterations:";
    AINFO << "   GPU usage: " << avg_gpu << "%";
    AINFO << "=========================================================";

    iteration_count_ = 0;
    total_gpu_usage_ = 0.0f;
  }

  return true;
}

CYBER_REGISTER_COMPONENT(GpuMonitorComponent)