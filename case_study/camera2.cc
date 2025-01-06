#include "cyber/examples/bachelor_thesis/case_study/camera2.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"

bool camera2_timer_component::Init() {
  // Create Writer that publishes to "/apollo/traffic_prediction"
  driver_writer_ = node_->CreateWriter<Driver>("/apollo/traffic_prediction");
  AINFO << "camera2_timer_component initialized.";
  return true;
}

bool camera2_timer_component::Proc() {
  static int seq = 0;
  const double start_time = apollo::cyber::Time::MonoTime().ToSecond();

  auto out_msg = std::make_shared<Driver>();
  out_msg->set_msg_id(seq++);
  out_msg->set_timestamp(start_time);

  AINFO << "================================================";
  AINFO << "[Chain2] Camera2 Timer triggers at time: " << start_time << "s";

  driver_writer_->Write(out_msg);
  return true;
}