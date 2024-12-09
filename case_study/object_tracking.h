#include <memory>

#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"


using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;
using apollo::cyber::Time;
class object_tracking_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;
  Time t1;
//  private:
//   std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(object_tracking_component)