#include "cyber/case_study/camera2.h"

#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"


bool camera2_timer_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/depth_estimation");
  return true;
}

bool camera2_timer_component::Proc() {
  // static int count = 0; 
  // if (count >= 100) {
  //     return false;
  // }
  auto start_time = t2.MonoTime().ToSecond();
  AINFO << "The start time of task chain 2 is " << start_time << "s";
  static int i = 0;
  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(i++);
  out_msg->set_timestamp(start_time);
  driver_writer_->Write(out_msg);
  // AINFO << "camera1_timer_component: Write drivermsg->"
  //       << out_msg->ShortDebugString();
  
  return true;
}