#include <memory>
#include "cyber/cyber.h"
#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/component/timer_component.h"
#include "cyber/examples/proto/examples.pb.h"

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::TimerComponent;
using apollo::cyber::Writer;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Time;
class camera1_timer_component : public TimerComponent {
 public:
  bool Init() override;
  bool Proc() override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
  Time t1;
};

CYBER_REGISTER_COMPONENT(camera1_timer_component)