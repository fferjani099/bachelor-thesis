#include "cyber/case_study/camera1.h"

#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"


bool camera1_timer_component::Init() {
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/pre_processing");
  return true;
}

bool camera1_timer_component::Proc() {
  apollo::cyber::Time mono_time = apollo::cyber::Time::MonoTime();
  auto t1 = mono_time.ToSecond();
  static int i = 0;
  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(i++);
  out_msg->set_timestamp(t1);
  // out_msg->set_timestamp(t1.Now().ToMicrosecond);
  AINFO << "================================================";
  AINFO << "The start time of task chain 1 is " << t1 << "s";
  driver_writer_->Write(out_msg);
  // AINFO << "camera1_timer_component: Write drivermsg->"
  //       << out_msg->ShortDebugString();
  
  return true;
}