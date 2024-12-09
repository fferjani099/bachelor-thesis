#include "cyber/case_study/pre_processing.h"
// #include <bits/stdc++.h>
// #include "cyber/croutine/croutine.h"
// using apollo::cyber::croutine::CRoutine;
// #include "torch/torch.h"
// #include "torch/script.h"
// #include "opencv2/opencv.hpp"
bool pre_pocessing_component::Init() {
  AINFO << "pre_pocessing_component init";
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/object_detection");
  return true;
}

bool pre_pocessing_component::Proc(const std::shared_ptr<Driver>& msg0) {
    static int i = 0;
    // AINFO << "Start pre_pocessing_component Proc [" << msg0->msg_id() << "]";
    // AINFO << "Start pre_pocessing_component Proc at [" << msg0->timestamp() << "]";
    auto out_msg = std::make_shared<Driver>();
    out_msg->set_msg_id(i++);
    out_msg->set_timestamp(msg0->timestamp());
    driver_writer_->Write(out_msg);
    return true;
}