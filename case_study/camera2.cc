#include "cyber/examples/bachelor_thesis/case_study/camera2.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"


bool camera2_timer_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/traffic_prediction");
  return true;
}

bool camera2_timer_component::Proc() {
  apollo::cyber::Time mono_time = apollo::cyber::Time::MonoTime();
  auto t2 = mono_time.ToSecond();
  static int i = 0;
  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(i++);
  out_msg->set_timestamp(t2);
  AINFO << "================================================";
  AINFO << "The start time of task chain 2 is " << t2 << "s";
  driver_writer_->Write(out_msg);
  
  return true;
}