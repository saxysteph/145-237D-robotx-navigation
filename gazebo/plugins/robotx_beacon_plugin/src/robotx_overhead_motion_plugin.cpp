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
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <sdf/Element.hh>

namespace robotx
{
/// Drives ``uav_dataset_camera`` on an overhead elliptical path + slow altitude wobble.
/// Uses WorldPoseCmd each step (model should be non-static; link gravity off).
class OverheadMotionPlugin :
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
      gzerr << "[OverheadMotionPlugin] Invalid model entity." << std::endl;
      return;
    }

    if (_sdf)
    {
      if (_sdf->HasElement("radius_x_m"))
        this->radiusX = std::max(0.0, _sdf->Get<double>("radius_x_m"));
      if (_sdf->HasElement("radius_y_m"))
        this->radiusY = std::max(0.0, _sdf->Get<double>("radius_y_m"));
      if (_sdf->HasElement("angular_speed_rad_s"))
        this->omega = _sdf->Get<double>("angular_speed_rad_s");
      if (_sdf->HasElement("phase_rad"))
        this->phase = _sdf->Get<double>("phase_rad");
      if (_sdf->HasElement("z_wobble_m"))
        this->zWobble = std::max(0.0, _sdf->Get<double>("z_wobble_m"));
      if (_sdf->HasElement("z_wobble_hz"))
        this->zWobbleHz = std::max(0.0, _sdf->Get<double>("z_wobble_hz"));
      if (_sdf->HasElement("z_wobble_phase_rad"))
        this->zPhase = _sdf->Get<double>("z_wobble_phase_rad");
    }

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

    const double c = std::cos(this->omega * t + this->phase);
    const double s = std::sin(this->omega * t + this->phase);

    gz::math::Pose3d pose = this->basePose;
    pose.Pos().X(this->basePose.Pos().X() + this->radiusX * c);
    pose.Pos().Y(this->basePose.Pos().Y() + this->radiusY * s);
    if (this->zWobble > 1e-6 && this->zWobbleHz > 1e-6)
    {
      const double wz = 2.0 * M_PI * this->zWobbleHz;
      pose.Pos().Z(this->basePose.Pos().Z() + this->zWobble * std::sin(wz * t + this->zPhase));
    }

    // Keep model-level yaw fixed (nadir stability); sensor child pose handles look-down.
    pose.Rot() = this->basePose.Rot();

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
  double radiusX{6.5};
  double radiusY{5.0};
  double omega{0.07};
  double phase{0.0};
  double zWobble{0.25};
  double zWobbleHz{0.12};
  double zPhase{0.0};
};
}  // namespace robotx

GZ_ADD_PLUGIN(
  robotx::OverheadMotionPlugin,
  gz::sim::System,
  gz::sim::ISystemConfigure,
  gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(robotx::OverheadMotionPlugin, "robotx::OverheadMotionPlugin")
