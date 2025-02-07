#pragma once

#include <nvml.h>
#include "cyber/cyber.h"
#include "cyber/component/timer_component.h"

/**
 * A simple Cyber TimerComponent that queries the NVIDIA GPU usage once per T
 * and logs the average usage over 100 iterations.
 */
class GpuMonitorComponent : public apollo::cyber::TimerComponent {
 public:
  bool Init() override;
  bool Proc() override;

 private:
  bool nvml_initialized_ = false;
  nvmlDevice_t device_;

  int iteration_count_ = 0;
  float total_gpu_usage_ = 0.0f;
};