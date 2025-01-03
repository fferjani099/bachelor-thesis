#include "cyber/examples/bachelor_thesis/case_study/traffic_prediction.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
bool traffic_prediction_component::Init() {
  AINFO << "depth_estimation_component init";
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
    cv::Mat image;

    switch (i % 4) {
        case 0:
            image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/0_green/0.png");
            break;
        case 1:
            image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/1_red/0.jpg");
            break;
        case 2:
            image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/2_yellow/0.jpg");
            break;
        case 3:
            image = cv::imread("/apollo/cyber/examples/bachelor_thesis/case_study/test_images/3_off/0.png");
            break;
        default:
            break;
    }
  
  
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    uchar* img = image.data;
    
    torch::Device device(torch::kCUDA);
    torch::Tensor img_tensor = torch::from_blob(img, {1, image.rows, image.cols, 3}, torch::kByte).to(device);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255.0);

    torch::jit::script::Module net = torch::jit::load("/apollo/cyber/examples/bachelor_thesis/case_study/resnet50_trace.pt");
    
    net.to(device);
    auto start_gpu = t1.MonoTime().ToSecond();
    torch::NoGradGuard no_grad;
    torch::Tensor output = net.forward({img_tensor}).toTensor();
    
    auto end_gpu = t1.MonoTime().ToSecond();
    AINFO << "The GPU start time is " << start_gpu << "s and the end time is " << end_gpu<< "s.";
    double inference_time = (end_gpu - start_gpu) * 1000; 
    AINFO << "The total time for model inference is " << inference_time << " ms.";
    total_gpu_time += inference_time;
    auto end_time = t1.MonoTime().ToSecond();    
    AINFO << "The end time of task chain 2 is "<< end_time  << "s";
    count ++;
    return true;
}