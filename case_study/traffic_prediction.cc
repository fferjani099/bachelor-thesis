#include "cyber/examples/bachelor_thesis/case_study/traffic_prediction.h"
#include <cuda_runtime.h>
#include "torch/torch.h"
#include "opencv2/opencv.hpp"

bool traffic_prediction_component::Init() {
  AINFO << "[traffic_prediction_component] Init start.";
  try {
    model_ = torch::jit::load("/apollo/cyber/examples/bachelor_thesis/case_study/resnet50_trace.pt");
    model_.to(torch::kCUDA);
    model_.eval();
    model_loaded_ = true;
    AINFO << "[traffic_prediction_component] Model loaded successfully.";
  } catch (const c10::Error& e) {
    AERROR << "[traffic_prediction_component] Error loading the model: " << e.what();
    model_loaded_ = false;
  }
  return model_loaded_;
}

bool traffic_prediction_component::Proc(const std::shared_ptr<Driver>& msg0) {
  if (!model_loaded_) {
    AERROR << "[traffic_prediction_component] Model not loaded. Skipping inference.";
    return false;
  }

  cv::Mat image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/traffic_light_view.png");
  if (image.empty()) {
    AERROR << "[traffic_prediction_component] Could not load traffic_light_view.png.";
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  torch::NoGradGuard no_grad;
  torch::Device device(torch::kCUDA);

  torch::Tensor img_tensor = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
  img_tensor = img_tensor.permute({0, 3, 1, 2}).toType(torch::kFloat).div(255.0);

  // Measure GPU time with CUDA events
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);
  auto output = model_.forward({img_tensor}).toTensor();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  float inference_time_ms = 0.0f;
  cudaEventElapsedTime(&inference_time_ms, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  if (skip_count_ > 0) {
    skip_count_--;
    AINFO << "[Chain1 - Object Detection] First inference (or warm-up) took ~"
          << inference_time_ms << " ms (GPU). Not counting towards average.";
    return true;
  }

  inference_count_++;
  total_gpu_time_ms_ += inference_time_ms;

  if (inference_count_ == 100) {
    double avg_time = total_gpu_time_ms_ / 100.0;
    AINFO << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    AINFO << "[Chain2 - Traffic Prediction] Average GPU time over 100 inferences: "
          << avg_time << " ms";
    AINFO << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    inference_count_ = 0;
    total_gpu_time_ms_ = 0.0;
  }

  const double end_time = apollo::cyber::Time::MonoTime().ToSecond();
  AINFO << "[Chain2 - Traffic Prediction] Single inference took ~"
        << inference_time_ms << " ms (GPU). Camera2 timestamp: "
        << msg0->timestamp() << ", end time: " << end_time << "s";

  return true;
}