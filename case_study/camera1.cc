#include "cyber/examples/bachelor_thesis/case_study/camera1.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"

bool camera1_timer_component::Init() {
  // Create Writer that publishes to "/apollo/object_detection"
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/object_detection");
  AINFO << "camera1_timer_component initialized.";
  return true;
}

bool camera1_timer_component::Proc() {
  // Publish a simple message
  static int seq = 0;
  const double start_time = apollo::cyber::Time::MonoTime().ToSecond();

  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(seq++);
  out_msg->set_timestamp(start_time);

  AINFO << "================================================";
  AINFO << "[Chain1] Camera1 Timer triggers at time: " << start_time << "s";

  driver_writer_->Write(out_msg);
  return true;
}