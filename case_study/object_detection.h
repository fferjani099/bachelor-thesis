#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"

// This component receives messages from /apollo/object_detection (camera1 output),
// performs a mock "object detection" forward pass on ResNet-50, then publishes 
// results and logs inference timing.

using apollo::cyber::Component;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;

class object_detection_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;

  torch::jit::script::Module model_;
  bool model_loaded_ = false;

  int inference_count_ = 0;
  double total_gpu_time_ms_ = 0.0;
  int skip_count_ = 2;
};

CYBER_REGISTER_COMPONENT(object_detection_component)
