#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include <gz/common/Console.hh>
#include <gz/msgs/light.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Light.hh>
#include <gz/sim/components/LightCmd.hh>
#include <gz/sim/components/Name.hh>
#include <sdf/Element.hh>
#include <sdf/Light.hh>

namespace robotx
{
enum class BeaconMode
{
  FlashRed,
  FlashGreen,
  FlashBlue,
  SteadyBlue,
  Off
};

class BeaconFlashPlugin :
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "[BeaconFlashPlugin] Invalid model entity." << std::endl;
      return;
    }

    if (_sdf && _sdf->HasElement("light_name"))
      this->lightName = _sdf->Get<std::string>("light_name");

    if (_sdf && _sdf->HasElement("mode"))
      this->mode = this->ParseMode(_sdf->Get<std::string>("mode"));

    if (_sdf && _sdf->HasElement("period_sec"))
      this->periodSec = std::max(0.1, _sdf->Get<double>("period_sec"));

    if (_sdf && _sdf->HasElement("off_intensity_scale"))
      this->offScale = std::max(0.0, std::min(1.0, _sdf->Get<double>("off_intensity_scale")));

    this->lightEntity = this->FindLightByName(_ecm, this->lightName);
    if (gz::sim::kNullEntity == this->lightEntity)
    {
      gzerr << "[BeaconFlashPlugin] Failed to find light entity named ["
            << this->lightName << "] in model [" << this->model.Name(_ecm) << "]." << std::endl;
    }
  }

  void PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || gz::sim::kNullEntity == this->lightEntity)
      return;

    const double simSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(_info.simTime).count();

    bool isOn = true;
    switch (this->mode)
    {
      case BeaconMode::FlashRed:
      case BeaconMode::FlashGreen:
      case BeaconMode::FlashBlue:
      {
        // 1s ON / 1s OFF when periodSec=1.0.
        const int phase = static_cast<int>(std::floor(simSec / this->periodSec));
        isOn = (phase % 2 == 0);
        break;
      }
      case BeaconMode::SteadyBlue:
        isOn = true;
        break;
      case BeaconMode::Off:
        isOn = false;
        break;
    }

    gz::msgs::Light cmd;
    cmd.set_name(this->lightName);

    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (this->mode)
    {
      case BeaconMode::FlashRed:
        r = 1.0f; g = 0.0f; b = 0.0f; break;
      case BeaconMode::FlashGreen:
        r = 0.0f; g = 1.0f; b = 0.0f; break;
      case BeaconMode::FlashBlue:
      case BeaconMode::SteadyBlue:
        r = 0.0f; g = 0.0f; b = 1.0f; break;
      case BeaconMode::Off:
        r = 0.0f; g = 0.0f; b = 0.0f; break;
    }

    const float s = isOn ? 1.0f : static_cast<float>(this->offScale);
    auto *diff = cmd.mutable_diffuse();
    diff->set_r(r * s);
    diff->set_g(g * s);
    diff->set_b(b * s);
    diff->set_a(1.0f);

    auto *spec = cmd.mutable_specular();
    spec->set_r(r * s);
    spec->set_g(g * s);
    spec->set_b(b * s);
    spec->set_a(1.0f);

    auto *existing = _ecm.Component<gz::sim::components::LightCmd>(this->lightEntity);
    if (!existing)
    {
      _ecm.CreateComponent(this->lightEntity, gz::sim::components::LightCmd(cmd));
    }
    else
    {
      existing->SetData(cmd, [](const gz::msgs::Light &, const gz::msgs::Light &) { return false; });
    }
  }

private:
  BeaconMode ParseMode(const std::string &_mode) const
  {
    if (_mode == "flash_red") return BeaconMode::FlashRed;
    if (_mode == "flash_green") return BeaconMode::FlashGreen;
    if (_mode == "flash_blue") return BeaconMode::FlashBlue;
    if (_mode == "steady_blue") return BeaconMode::SteadyBlue;
    if (_mode == "off") return BeaconMode::Off;
    return BeaconMode::FlashRed;
  }

  gz::sim::Entity FindLightByName(
    const gz::sim::EntityComponentManager &_ecm,
    const std::string &_name) const
  {
    gz::sim::Entity out = gz::sim::kNullEntity;
    _ecm.Each<gz::sim::components::Name, gz::sim::components::Light>(
      [&](const gz::sim::Entity &_entity,
          const gz::sim::components::Name *_n,
          const gz::sim::components::Light *) -> bool
      {
        if (_n && _n->Data() == _name)
        {
          out = _entity;
          return false;
        }
        return true;
      });
    return out;
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Entity lightEntity{gz::sim::kNullEntity};
  std::string lightName{"top_beacon"};
  BeaconMode mode{BeaconMode::FlashRed};
  double periodSec{1.0};
  double offScale{0.02};
};
}  // namespace robotx

GZ_ADD_PLUGIN(
  robotx::BeaconFlashPlugin,
  gz::sim::System,
  gz::sim::ISystemConfigure,
  gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(robotx::BeaconFlashPlugin, "robotx::BeaconFlashPlugin")
