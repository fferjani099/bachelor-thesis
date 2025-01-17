#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"

// This component receives messages from /apollo/traffic_prediction (camera2 output),
// performs a mock "traffic prediction" forward pass on ResNet-50, then logs timing.

using apollo::cyber::Component;
using apollo::cyber::examples::proto::Driver;

class traffic_prediction_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;

 private:
  torch::jit::script::Module model_;
  bool model_loaded_ = false;

  int inference_count_ = 0;
  double total_gpu_time_ms_ = 0.0;
  int skip_count_ = 1;
};

CYBER_REGISTER_COMPONENT(traffic_prediction_component)
