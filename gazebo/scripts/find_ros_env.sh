#!/usr/bin/env bash
# Print likely ROS 2 setup.bash locations on this machine (run from host where you use ros2).

set +u
echo "=== ros2 on PATH ==="
command -v ros2 2>/dev/null || echo "(none)"
type ros2 2>/dev/null || true

echo ""
echo "=== /opt/ros/*/setup.bash ==="
if [[ -d /opt/ros ]]; then
  ls /opt/ros 2>/dev/null
  for d in /opt/ros/*/setup.bash; do
    [[ -f "$d" ]] && echo "  $d"
  done
else
  echo "  (no /opt/ros — typical on macOS without Linux ROS debs)"
fi

echo ""
echo "=== colcon workspaces: */install/setup.bash under \$HOME (depth ≤6) ==="
find "${HOME}" -maxdepth 6 -path '*/install/setup.bash' 2>/dev/null | while read -r p; do
  if head -5 "$p" 2>/dev/null | grep -qE 'AMENT_PREFIX_PATH|ROS_DISTRO'; then
    echo "  $p"
  fi
done

echo ""
echo "=== common VM / shared folders ==="
for m in /Volumes/Ubuntu /mnt/wsl /Users/*/ubuntu*; do
  [[ -e "$m" ]] && {
    echo "  checking $m ..."
    find "$m" -maxdepth 5 -name 'setup.bash' -path '*/opt/ros/*' 2>/dev/null | head -5
  }
done 2>/dev/null

echo ""
echo "If a line appears above, use it as ROS_SETUP_BASH, e.g.:"
echo "  export ROS_SETUP_BASH=/that/path/setup.bash"
echo "  export REPO_ROOT=\$PWD && source \"\$REPO_ROOT/gazebo/scripts/setup_robotx_sim_env.sh\""
