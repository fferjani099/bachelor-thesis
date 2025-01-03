#include "cyber/examples/bachelor_thesis/case_study/object_tracking.h"
// #include <bits/stdc++.h>
// #include "cyber/croutine/croutine.h"
// using apollo::cyber::croutine::CRoutine;
// #include "torch/torch.h"
// #include "torch/script.h"
// #include "opencv2/opencv.hpp"
bool object_tracking_component::Init() {
  AINFO << "object_tracking_component init";
//   driver_writer_ = node_->CreateWriter<Driver>("/apollo/object_tracking");
  return true;
}

bool object_tracking_component::Proc(const std::shared_ptr<Driver>& msg0) {
    // AINFO << "Start object_tracking_component Proc [" << msg0->msg_id() << "]";
    auto end_time = t1.MonoTime().ToSecond();
    AINFO << "The end time of task chain 1 is "<< end_time  << "s";
    // auto start_time = msg0->timestamp();
    // AINFO << "The time of the chain 1 is " << (end_time-start_time)*1000 <<" ms";
    // auto out_msg = std::make_shared<Driver>();
    // out_msg->set_msg_id(i++);
    // driver_writer_->Write(out_msg);
    return true;
}