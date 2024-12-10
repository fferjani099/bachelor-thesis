#include "cyber/case_study/depth_estimation.h"
// #include <bits/stdc++.h>
// #include "cyber/croutine/croutine.h"
// using apollo::cyber::croutine::CRoutine;
// #include "torch/torch.h"
// #include "torch/script.h"
// #include "opencv2/opencv.hpp"
bool depth_estimation_component::Init() {
  AINFO << "depth_estimation_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/traffic_prediction");
  return true;
}

bool depth_estimation_component::Proc(const std::shared_ptr<Driver>& msg0) {
    static int i = 0;
    // AINFO << "Start depth_estimation_component Proc [" << msg0->msg_id() << "]";
    
    auto out_msg = std::make_shared<Driver>();
    out_msg->set_msg_id(i++);
    out_msg->set_timestamp(msg0->timestamp());
    driver_writer_->Write(out_msg);
    return true;
}