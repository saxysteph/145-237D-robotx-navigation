#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/sim/components/Pose.hh>
#include <sdf/Element.hh>

namespace robotx
{
class BobbingPlugin :
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
      gzerr << "[BobbingPlugin] Invalid model entity." << std::endl;
      return;
    }

    if (_sdf && _sdf->HasElement("amplitude_z_m"))
      this->ampZ = std::max(0.0, _sdf->Get<double>("amplitude_z_m"));
    if (_sdf && _sdf->HasElement("frequency_hz"))
      this->freqHz = std::max(0.01, _sdf->Get<double>("frequency_hz"));
    if (_sdf && _sdf->HasElement("drift_xy_m"))
      this->driftXY = std::max(0.0, _sdf->Get<double>("drift_xy_m"));
    if (_sdf && _sdf->HasElement("drift_frequency_hz"))
      this->driftHz = std::max(0.001, _sdf->Get<double>("drift_frequency_hz"));
    if (_sdf && _sdf->HasElement("roll_deg"))
      this->rollDeg = std::max(0.0, _sdf->Get<double>("roll_deg"));
    if (_sdf && _sdf->HasElement("pitch_deg"))
      this->pitchDeg = std::max(0.0, _sdf->Get<double>("pitch_deg"));
    if (_sdf && _sdf->HasElement("phase_rad"))
      this->phase = _sdf->Get<double>("phase_rad");

    auto poseComp = _ecm.Component<gz::sim::components::Pose>(this->model.Entity());
    if (poseComp)
    {
      this->basePose = poseComp->Data();
      this->hasBasePose = true;
    }
  }

  void PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || !this->model.Valid(_ecm))
      return;

    if (!this->hasBasePose)
    {
      auto poseComp = _ecm.Component<gz::sim::components::Pose>(this->model.Entity());
      if (!poseComp)
        return;
      this->basePose = poseComp->Data();
      this->hasBasePose = true;
    }

    const double t =
      std::chrono::duration_cast<std::chrono::duration<double>>(_info.simTime).count();
    const double w = 2.0 * M_PI * this->freqHz;
    const double wd = 2.0 * M_PI * this->driftHz;
    const double s = std::sin(w * t + this->phase);
    const double c = std::cos(w * t + this->phase);

    gz::math::Pose3d pose = this->basePose;
    pose.Pos().X(this->basePose.Pos().X() + this->driftXY * std::sin(wd * t + this->phase));
    pose.Pos().Y(this->basePose.Pos().Y() + this->driftXY * 0.6 * std::cos(wd * t + this->phase));
    pose.Pos().Z(this->basePose.Pos().Z() + this->ampZ * s);

    const double deg2rad = M_PI / 180.0;
    const double roll = (this->rollDeg * deg2rad) * c;
    const double pitch = (this->pitchDeg * deg2rad) * s;
    const double yaw = this->basePose.Rot().Yaw();
    pose.Rot() = gz::math::Quaterniond(roll, pitch, yaw);

    auto poseCmd = _ecm.Component<gz::sim::components::WorldPoseCmd>(this->model.Entity());
    if (!poseCmd)
      _ecm.CreateComponent(this->model.Entity(), gz::sim::components::WorldPoseCmd(pose));
    else
      poseCmd->SetData(pose, [](const auto &, const auto &) { return false; });
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  bool hasBasePose{false};
  gz::math::Pose3d basePose{0, 0, 0, 0, 0, 0};
  double ampZ{0.06};
  double freqHz{0.20};
  double driftXY{0.03};
  double driftHz{0.035};
  double rollDeg{1.2};
  double pitchDeg{1.4};
  double phase{0.0};
};
}  // namespace robotx

GZ_ADD_PLUGIN(
  robotx::BobbingPlugin,
  gz::sim::System,
  gz::sim::ISystemConfigure,
  gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(robotx::BobbingPlugin, "robotx::BobbingPlugin")
