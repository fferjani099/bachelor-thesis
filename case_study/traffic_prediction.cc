#include "cyber/case_study/traffic_prediction.h"
// #include <bits/stdc++.h>
// #include "cyber/croutine/croutine.h"
// using apollo::cyber::croutine::CRoutine;
#include "torch/torch.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
bool traffic_prediction_component::Init() {
  AINFO << "depth_estimation_component init";
//   driver_writer_ = node_->CreateWriter<Driver>("/apollo/depth_estimation");
  return true;
}

bool traffic_prediction_component::Proc(const std::shared_ptr<Driver>& msg0) {
    static int count = 0; 
    static double total_gpu_time = 0.0;  

    if (count >= 100) {
        double average_gpu_time = total_gpu_time / 100;
        AINFO << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
        AINFO << "Average GPU time for 100 inferences in chain 2: " << average_gpu_time << " ms";
        AINFO << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
        count = 0; 
    }
    static int i = 0;
    // AINFO << "Start traffic_prediction_component Proc [" << msg0->msg_id() << "]";
    cv::Mat image;

    switch (i % 4) {
        case 0:
            image = cv::imread("/apollo/cyber/case_study/test_images/0_green/0.png");
            // AINFO << "0_green/0.png";
            break;
        case 1:
            image = cv::imread("/apollo/cyber/case_study/test_images/1_red/0.jpg");
            // AINFO << "1_red/0.jpg";
            break;
        case 2:
            image = cv::imread("/apollo/cyber/case_study/test_images/2_yellow/0.jpg");
            // AINFO << "2_yellow/0.jpg";
            break;
        case 3:
            image = cv::imread("/apollo/cyber/case_study/test_images/3_off/0.png");
            // AINFO << "3_off/0.jpg";
            break;
        default:
            // AINFO << "detect_data";
            break;
    }
  
  
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    uchar* img = image.data;
    
    torch::Device device(torch::kCUDA);
    torch::Tensor img_tensor = torch::from_blob(img, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255.0);

    torch::jit::script::Module net = torch::jit::load("/apollo/cyber/case_study/resnet50_256_trace.pt");
    
    net.to(device);
    // auto start_gpu = t1.MonoTime().ToMicrosecond();
    auto start_gpu = t1.MonoTime().ToSecond();
    torch::NoGradGuard no_grad;
    torch::Tensor output = net.forward({img_tensor}).toTensor();
    
    auto end_gpu = t1.MonoTime().ToSecond();
    AINFO << "The GPU start time is " << start_gpu << "s and the end time is " << end_gpu<< "s.";
    double inference_time = (end_gpu - start_gpu) * 1000; 
    AINFO << "The total time for model inference is " << inference_time << " ms.";
    total_gpu_time += inference_time;  // 累积GPU时间
    auto end_time = t1.MonoTime().ToSecond();
    // auto start_time = msg0->timestamp();


    // nvmlReturn_t result;
    // nvmlDevice_t n_device;
    // nvmlUtilization_t utilization;
    //  NVML
    // result = nvmlInit();
    // if (NVML_SUCCESS != result) {
    //     std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

    // unsigned int deviceCount;
    // result = nvmlDeviceGetCount(&deviceCount);
    // if (NVML_SUCCESS != result) {
    //     std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
    //     nvmlShutdown();
    //     return 1;
    // }

    // AINFO << "Number of GPUs: " << deviceCount ;

    // result = nvmlDeviceGetHandleByIndex(0, &n_device);
    

    // // Get utilization rates for the device
    // result = nvmlDeviceGetUtilizationRates(n_device, &utilization);
    
    // AINFO << "  GPU Utilization: " << utilization.gpu << "%" ;
    // AINFO << "  Memory Utilization: " << utilization.memory << "%" ;
    
    
    AINFO << "The end time of task chain 2 is "<< end_time  << "s";
    // auto out_msg = std::make_shared<Driver>();
    // out_msg->set_msg_id(i++);
    // driver_writer_->Write(out_msg);
    count ++;
    return true;
}