#include "cyber/examples/bachelor_thesis/case_study/object_detection.h"
#include "torch/torch.h"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>

bool object_detection_component::Init() {
  AINFO << "[object_detection_component] Init start.";

  driver_writer_ = node_->CreateWriter<Driver>("/apollo/object_tracking");

  try {
    model_ = torch::jit::load("/apollo/cyber/examples/bachelor_thesis/case_study/resnet50_trace.pt");
    model_.to(torch::kCUDA);
    model_.eval();
    model_loaded_ = true;
    AINFO << "[object_detection_component] Model loaded successfully.";
  } catch (const c10::Error& e) {
    AERROR << "[object_detection_component] Error loading the model: " << e.what();
    model_loaded_ = false;
  }

  return model_loaded_;
}

bool object_detection_component::Proc(const std::shared_ptr<Driver>& msg0) {
  if (!model_loaded_) {
    AERROR << "[object_detection_component] Model not loaded. Skipping inference.";
    return false;
  }

  cv::Mat image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/street_view.png");
  if (image.empty()) {
    AERROR << "[object_detection_component] Could not load /apollo/.../street_view.png";
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  torch::NoGradGuard no_grad;
  torch::Device device(torch::kCUDA);
  torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
  img_tensor = img_tensor.permute({0, 3, 1, 2}).toType(torch::kFloat).div(255.0);

  // Measure GPU time with CUDA events
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);
  auto outputs = model_.forward({img_tensor}).toTensor();
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
    AINFO << "[Chain1 - Object Detection] Average GPU time over 100 inferences: "
          << avg_time << " ms";
    AINFO << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    inference_count_ = 0;
    total_gpu_time_ms_ = 0.0;
  }

  AINFO << "[Chain1 - Object Detection] Single inference took ~"
        << inference_time_ms << " ms (GPU). Timestamp from camera1: "
        << msg0->timestamp();

  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(msg0->msg_id());
  out_msg->set_timestamp(msg0->timestamp());

  return true;
}