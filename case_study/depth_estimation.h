#include <memory>

#include "cyber/component/component.h"
#include "cyber/examples/proto/examples.pb.h"


using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Driver;
using apollo::cyber::Writer;
class depth_estimation_component : public Component<Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0) override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(depth_estimation_component)