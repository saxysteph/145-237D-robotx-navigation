#!/usr/bin/env python3
"""
Record camera frames from Gazebo *without* ROS: subscribe to gz.msgs.Image on gz-transport.

Replaces ros_gz_bridge + record_uav_dataset.py for data collection. Run while `gz sim` is
active with the same GZ_SIM_* env as gz_env.sh.

Dependencies (must match Homebrew Gazebo’s Python bindings, usually 3.12+ on macOS):
  pip install "protobuf>=4.0" numpy opencv-python
  plus PYTHONPATH to gz-transport / gz-msgs (see --help).

Linux (apt): often `python3-gz-msgs` / transport packages put modules on sys.path automatically.

Writes the same filenames and uav_pose_log.csv layout as record_uav_dataset.py.

Subscribes to two gz.msgs.Pose_V streams when enabled:

- **--pose-dynamic-topic** (default ``…/dynamic_pose/info``): moving links (bobbing buoy
  ``body_link`` positions → ``buoy_poses_json``).
- **--pose-static-topic** (default ``…/pose/info``): includes **static** models such as
  ``uav_dataset_camera`` → ``drone_*`` columns.

Static rigs do **not** appear on ``dynamic_pose/info`` alone; that is why camera pose
would stay empty without ``pose/info``. A **moving** ``uav_dataset_camera`` (overhead orbit)
is published on ``dynamic_pose/info``; this recorder therefore updates ``drone_*`` from
**either** stream when a matching pose is present.

When clip ``world.sdf`` exists under ``--out-dir``, buoy model instance names are read and
their **live** XYZ is sampled from each ``Pose_V`` message (preferred link ``body_link``)
and written to column ``buoy_poses_json`` so ``project_buoys_to_yolo_labels`` can relate
camera + buoy poses per frame.

For a static UAV camera rig, drone columns may still be constant frame-to-frame; buoy JSON
captures XY drift/z bobbing from bobbing-plugin motion when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from pathlib import Path


def _ensure_dyld_library_path_macos() -> None:
    """Homebrew gz-transport Python .so needs libgz-transport in the loader path (before import)."""
    if sys.platform != "darwin":
        return
    import shutil
    import subprocess

    brew = shutil.which("brew")
    try:
        if brew:
            gt_lib = subprocess.check_output([brew, "--prefix", "gz-transport13"], text=True).strip() + "/lib"
            hb_lib = subprocess.check_output([brew, "--prefix"], text=True).strip() + "/lib"
        else:
            gt_lib = "/opt/homebrew/opt/gz-transport13/lib"
            hb_lib = "/opt/homebrew/lib"
    except (subprocess.CalledProcessError, FileNotFoundError):
        gt_lib = "/opt/homebrew/opt/gz-transport13/lib"
        hb_lib = "/opt/homebrew/lib"
    cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    extra = ":".join([p for p in (gt_lib, hb_lib) if os.path.isdir(p)])
    if not extra:
        return
    os.environ["DYLD_LIBRARY_PATH"] = extra + (":" + cur if cur else "")


def _prepend_homebrew_gz_python_paths() -> None:
    """Add typical Homebrew Cellar paths so `import gz.transport13` works."""
    import glob

    # Match this interpreter only; globbing python3.*/site-packages picks 3.13/3.14 and breaks ABI.
    py_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    transport_roots: list[str] = []
    msgs_roots: list[str] = []
    for root in (prefix, "/usr/local"):
        transport_roots.extend(
            glob.glob(f"{root}/Cellar/gz-transport13/*/lib/python{py_mm}/site-packages")
        )
        transport_roots.extend(
            glob.glob(f"{root}/Cellar/gz-transport14/*/lib/python{py_mm}/site-packages")
        )
        msgs_roots.extend(glob.glob(f"{root}/Cellar/gz-msgs10/*/lib/python"))
        msgs_roots.extend(glob.glob(f"{root}/Cellar/gz-msgs11/*/lib/python"))
    seen: set[str] = set()
    # Insert msgs paths first, then transport so gz-transport stays at the front of sys.path.
    for group in (msgs_roots, transport_roots):
        for c in group:
            if c in seen or not os.path.isdir(c):
                continue
            seen.add(c)
            sys.path.insert(0, c)


def _import_gz() -> tuple[object, object, object]:
    _prepend_homebrew_gz_python_paths()
    try:
        from gz.transport13 import Node  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Cannot import gz.transport13. Install Gazebo Harmonic (gz-transport13) and set "
            "PYTHONPATH to Homebrew site-packages, or run on Linux with ROS-free gz Python "
            f"packages. Underlying: {e}"
        ) from e
    try:
        from gz.msgs10.image_pb2 import Image, PixelFormatType  # type: ignore
    except ImportError:
        from gz.msgs11.image_pb2 import Image, PixelFormatType  # type: ignore
    return Node, Image, PixelFormatType


def _import_pose_v_msg() -> object:
    try:
        from gz.msgs10.pose_v_pb2 import Pose_V  # type: ignore
    except ImportError:
        from gz.msgs11.pose_v_pb2 import Pose_V  # type: ignore
    return Pose_V


def _quat_wxyz_to_yaw_deg(wx: float, wy: float, wz: float, ww: float) -> float:
    """World-frame yaw about +Z (degrees) from ignition-style quaternion x,y,z,w."""
    siny_cosp = 2.0 * (ww * wz + wx * wy)
    cosy_cosp = 1.0 - 2.0 * (wy * wy + wz * wz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _pose_stamp_ns(pose_msg) -> int:
    if pose_msg.header and pose_msg.header.stamp:
        sec = int(pose_msg.header.stamp.sec)
        nsec = int(pose_msg.header.stamp.nsec)
        return sec * 1_000_000_000 + nsec
    return 0


def _snapshot_models_xyz_pose_v(pose_v_msg, wanted_models: frozenset[str]) -> dict[str, tuple[float, float, float]]:
    """
    Map world model instance names → centroid-ish link world XYZ from gz.msgs.Pose_V.

    Prefers poses named ``<instance>::body_link`` (RobotX buoy models).
    """
    from collections import defaultdict

    by_model: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for ap in pose_v_msg.pose:
        raw = ap.name or ""
        if "::" in raw:
            model, suf = raw.split("::", 1)
        else:
            model, suf = raw, ""
        if model not in wanted_models:
            continue
        pos = ap.position
        by_model[model].append((suf.lower(), float(pos.x), float(pos.y), float(pos.z)))
    out: dict[str, tuple[float, float, float]] = {}
    for inst in wanted_models:
        rows = by_model.get(inst)
        if not rows:
            continue
        chosen = None
        for suf_str, px, py, pz in rows:
            if suf_str == "body_link":
                chosen = (px, py, pz)
                break
        if chosen is None:
            non_coll = [(s, x, y, z) for s, x, y, z in rows if "collision" not in s]
            pool = non_coll if non_coll else rows
            s, px, py, pz = sorted(pool, key=lambda t: (len(t[0]), t[0]))[0]
            chosen = (px, py, pz)
        out[inst] = chosen
    return out


def _pick_named_pose(pose_v_msg, substr: str):
    """First pose in the vector whose name contains ``substr`` (case-sensitive)."""
    best = None
    best_key = (99, "")
    for ap in pose_v_msg.pose:
        name = ap.name or ""
        if substr not in name:
            continue
        pref = 0 if name == substr else 1
        key = (pref, name)
        if best is None or key < best_key:
            best = ap
            best_key = key
    return best


def image_to_bgr(msg, pixel_fmt_cls) -> "object | None":
    import numpy as np

    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0 or not msg.data:
        return None
    step = int(msg.step) if msg.step else w * 3
    raw = memoryview(msg.data)
    fmt = int(msg.pixel_format_type)

    # Match gz.msgs.PixelFormatType (stable numeric codes).
    if fmt == pixel_fmt_cls.BGR_INT8:
        arr = np.frombuffer(raw, dtype=np.uint8)
        expected = step * h
        got = arr.size
        if got < expected:
            return None
        arr = arr[:expected].reshape((h, step))
        return np.ascontiguousarray(arr[:, : w * 3]).reshape((h, w, 3))
    if fmt == pixel_fmt_cls.RGB_INT8:
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = arr[: step * h].reshape((h, step))[:, : w * 3].reshape((h, w, 3))
        import cv2

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", default="/robotx/uav/camera/image_raw", help="Gazebo transport topic (gz.msgs.Image)")
    p.add_argument("--out-dir", default="captures/gazebo_uav_gz", help="Output directory for jpg + CSV")
    p.add_argument("--save-every", type=int, default=1, help="Save every Nth message")
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Exit after this many seconds (0 = run until Ctrl+C).",
    )
    p.add_argument(
        "--pose-pose-v-topic",
        type=str,
        default="/world/robotx_task1_uav_view/dynamic_pose/info",
        help="Backward alias: sets **dynamic** Pose_V topic (same as --pose-dynamic-topic).",
    )
    p.add_argument(
        "--pose-dynamic-topic",
        type=str,
        default="",
        help="Pose_V dynamic_pose/info (moving buoys → buoy_poses_json). Default: inherit --pose-pose-v-topic.",
    )
    p.add_argument(
        "--pose-static-topic",
        type=str,
        default="/world/robotx_task1_uav_view/pose/info",
        help="Pose_V pose/info for static camera model (→ drone_*). Empty disables.",
    )
    p.add_argument(
        "--pose-model-substr",
        type=str,
        default="uav_dataset_camera",
        help="Substring to match against Pose_V entry ``name`` for drone_x/y/z/yaw CSV columns.",
    )
    args = p.parse_args()
    dyn_t = (args.pose_dynamic_topic or "").strip()
    if not dyn_t:
        dyn_t = (args.pose_pose_v_topic or "").strip()
    static_t = (args.pose_static_topic or "").strip()

    _ensure_dyld_library_path_macos()

    try:
        import cv2
    except ImportError as e:
        print("Need opencv-python and numpy:", e, file=sys.stderr)
        return 1

    try:
        Node, Image, PixelFormatType = _import_gz()
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1

    Pose_V_cls = None
    pose_topics: list[tuple[str, str]] = []
    if dyn_t and static_t and dyn_t == static_t:
        pose_topics = [("combined", dyn_t)]
    else:
        if dyn_t:
            pose_topics.append(("dynamic", dyn_t))
        if static_t:
            pose_topics.append(("static", static_t))
    enable_pose_v = len(pose_topics) > 0
    if enable_pose_v:
        try:
            Pose_V_cls = _import_pose_v_msg()
        except ImportError as e:
            print(f"Pose sync disabled (cannot import Pose_V): {e}", file=sys.stderr)
            pose_topics = []
            Pose_V_cls = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_csv_path = out_dir / "uav_pose_log.csv"

    buoy_models: frozenset[str] = frozenset()
    world_stub = out_dir / "world.sdf"
    if world_stub.is_file():
        _script_here = Path(__file__).resolve().parent
        if str(_script_here) not in sys.path:
            sys.path.insert(0, str(_script_here))
        try:
            from render_gt_in_image_from_pose import parse_buoy_instances_world_sdf  # noqa: E402

            buoy_models = frozenset(nm for nm, *_rest in parse_buoy_instances_world_sdf(world_stub))
        except (OSError, ValueError, ImportError) as ex:
            print(f"Buoy snapshot disabled (cannot parse {world_stub}): {ex}", file=sys.stderr)
    pose_lock = threading.Lock()
    latest_pose: dict[str, object] = {
        "pose_stamp_ns": "",
        "x": "",
        "y": "",
        "z": "",
        "yaw_deg": "",
        "buoy_json": "{}",
    }

    state = {
        "counter": 0,
        "saved": 0,
        "stop": False,
        # Once moving camera pose is observed on dynamic_pose, avoid static pose overwrite.
        "camera_seen_dynamic": False,
        "camera_source_logged": False,
        "pose_wait_logged": False,
    }

    def _update_drone_pose_from_ap(ap) -> None:
        if ap is None:
            return
        ps = _pose_stamp_ns(ap)
        pos = ap.position
        o = ap.orientation
        yaw_deg = _quat_wxyz_to_yaw_deg(o.x, o.y, o.z, o.w)
        latest_pose["pose_stamp_ns"] = str(ps) if ps else ""
        latest_pose["x"] = f"{pos.x:.9g}"
        latest_pose["y"] = f"{pos.y:.9g}"
        latest_pose["z"] = f"{pos.z:.9g}"
        latest_pose["yaw_deg"] = f"{yaw_deg:.9g}"

    def on_dynamic_pose_v(msg) -> None:
        if state.get("stop"):
            return
        with pose_lock:
            if buoy_models:
                snap = _snapshot_models_xyz_pose_v(msg, buoy_models)
                latest_pose["buoy_json"] = json.dumps({k: [v[0], v[1], v[2]] for k, v in snap.items()})
            ap_cam = _pick_named_pose(msg, args.pose_model_substr)
            if ap_cam is not None:
                _update_drone_pose_from_ap(ap_cam)
                state["camera_seen_dynamic"] = True
                if not state["camera_source_logged"]:
                    print(f"Camera pose source: dynamic Pose_V ({dyn_t})")
                    state["camera_source_logged"] = True

    def on_static_pose_v(msg) -> None:
        if state.get("stop"):
            return
        # Dynamic camera pose (moving rig) is preferred when available.
        if state.get("camera_seen_dynamic"):
            return
        ap = _pick_named_pose(msg, args.pose_model_substr)
        if ap is None:
            return
        with pose_lock:
            _update_drone_pose_from_ap(ap)

    def on_combined_pose_v(msg) -> None:
        if state.get("stop"):
            return
        on_dynamic_pose_v(msg)
        on_static_pose_v(msg)

    with pose_csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "image_name",
                "image_stamp_ns",
                "pose_stamp_ns",
                "drone_x_m",
                "drone_y_m",
                "drone_z_m",
                "drone_yaw_deg",
                "navsat_stamp_ns",
                "drone_lat_deg",
                "drone_lon_deg",
                "drone_alt_m",
                "buoy_poses_json",
            ]
        )

    def on_image(msg) -> None:
        if state["stop"]:
            return
        if enable_pose_v:
            with pose_lock:
                have_pose = bool(str(latest_pose.get("x", "")).strip())
            if not have_pose:
                if not state["pose_wait_logged"]:
                    print("Waiting for first camera Pose_V before saving frames...")
                    state["pose_wait_logged"] = True
                return
        state["counter"] += 1
        if state["counter"] % args.save_every != 0:
            return
        bgr = image_to_bgr(msg, PixelFormatType)
        if bgr is None:
            print(f"Unsupported or short frame format={msg.pixel_format_type}", file=sys.stderr)
            return
        stamp_ns = 0
        if msg.header and msg.header.stamp:
            sec = int(msg.header.stamp.sec)
            nsec = int(msg.header.stamp.nsec)
            stamp_ns = sec * 1_000_000_000 + nsec
        if stamp_ns == 0:
            stamp_ns = int(time.time() * 1e9)
        name = f"uav_{stamp_ns}.jpg"
        path = out_dir / name
        if not cv2.imwrite(str(path), bgr):
            print(f"imwrite failed: {path}", file=sys.stderr)
            return
        state["saved"] += 1
        with pose_lock:
            prow = [
                latest_pose["pose_stamp_ns"],
                latest_pose["x"],
                latest_pose["y"],
                latest_pose["z"],
                latest_pose["yaw_deg"],
            ]
            bj = str(latest_pose["buoy_json"])
        with pose_csv_path.open("a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([name, stamp_ns, *prow, "", "", "", "", bj])
        if state["saved"] % 25 == 0:
            print(f"Saved {state['saved']} frames → {out_dir}")

    node = Node()
    node.subscribe(Image, args.topic, on_image)
    print(f"Listening (Gazebo Transport, no ROS): {args.topic}")
    if Pose_V_cls is not None and pose_topics:
        if pose_topics == [("combined", dyn_t)] or (
            len(pose_topics) == 1 and pose_topics[0][0] == "combined"
        ):
            _c, tp0 = pose_topics[0]
            node.subscribe(Pose_V_cls, tp0, on_combined_pose_v)
            print(f"Pose_V combined: {tp0}")
        elif len(pose_topics) == 2:
            node.subscribe(Pose_V_cls, pose_topics[0][1], on_dynamic_pose_v)
            print(f"Pose_V dynamic (buoys): {pose_topics[0][1]}")
            node.subscribe(Pose_V_cls, pose_topics[1][1], on_static_pose_v)
            print(f"Pose_V static (camera): {pose_topics[1][1]} (∋ {args.pose_model_substr!r})")
        else:
            _lbl, tp0 = pose_topics[0]
            if _lbl == "dynamic":
                node.subscribe(Pose_V_cls, tp0, on_combined_pose_v)
                print(f"Pose_V dynamic-only: {tp0} (enable --pose-static-topic for static camera)")
            else:
                node.subscribe(Pose_V_cls, tp0, on_static_pose_v)
                print(f"Pose_V static-only: {tp0} (∋ {args.pose_model_substr!r})")
        if buoy_models:
            print(f"Buoy XYZ sync: {len(buoy_models)} instance(s) from {world_stub.name}")
    if args.duration > 0:
        print(f"Writing to {out_dir} for {args.duration:.1f} s")
    else:
        print(f"Writing to {out_dir} (Ctrl+C to stop)")
    start = time.monotonic()
    try:
        while not state["stop"]:
            if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                state["stop"] = True
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        state["stop"] = True
    print(f"Finished. Saved {state['saved']} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
