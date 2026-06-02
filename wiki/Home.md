# RobotX Navigation — Aerial Intelligence & Communication Pipeline

> **CSE 145 / CSE 237D — Spring 2026 | UC San Diego**
> Advised by Prof. Jack Silberman | In collaboration with TritonAI & Team Inspiration

---

## Abstract

To successfully deploy an aerial-to-surface recognition system that extends the navigational intelligence of a surface vessel, the drone must integrate an edge-compute color thresholding model for obstacle detection, mathematical ray-projection logic for coordinate mapping, and a low-latency MAVLink publish-subscribe network for actionable payload transmission.

In plain terms: a drone acts as an overhead scout, autonomously identifying colored buoys, calculating their real-world GPS positions, and transmitting that map to a surface vehicle — enabling the surface vehicle to navigate a course it cannot see on its own.

This project contributes the **aerial perception and communication layer** of UCSD's entry into the [Maritime RobotX 2026 Challenge](https://robotx.org/).

---

## Quick Links

| Resource | Link |
|---|---|
| GitHub Repository | [saxysteph/145-237D-robotx-navigation](https://github.com/saxysteph/145-237D-robotx-navigation) |
| Partner Run Instructions | [fulldemo/PARTNER_INSTRUCTIONS.md](../blob/main/fulldemo/PARTNER_INSTRUCTIONS.md) |
| Full Demo Guide | [fulldemo/README.md](../blob/main/fulldemo/README.md) |
| Team | [Team](Team) |
| Project Overview | [Project-Overview](Project-Overview) |
| Repository Structure | [Repository-Structure](Repository-Structure) |
| Setup & Replication | [Setup-and-Replication](Setup-and-Replication) |

---

## System Overview

```
┌─────────────────────────────────┐
│         UAV (Drone)             │
│  Camera → YOLO → HSV → GPS Map │
│       Jetson Orin Nano          │
└────────────┬────────────────────┘
             │ MAVLink UDP
             ▼
┌─────────────────────────────────┐
│      Ground Station / USV       │
│  Receives buoy GPS coordinates  │
└─────────────────────────────────┘
```

The UAV runs a two-stage perception pipeline on a Jetson Orin Nano:
1. **YOLO** (`buoy_best.onnx`) proposes bounding boxes around buoy candidates
2. **HSV thresholding** classifies each ROI as red, green, or blue
3. **Ray projection** converts pixel centroids to GPS coordinates using drone altitude, heading, and camera intrinsics
4. **MAVLink STATUSTEXT** packets transmit each detection to the ground station over UDP

---

## Project Status

| Milestone | Priority | Status |
|---|---|---|
| **Wk 4 — System Setup:** Jetson Orin setup, live camera feed, basic detection running | Necessary | ✅ Complete |
| **Wk 5 — MVP:** Buoy color detection, GPS coordinate projection, MAVLink transmission to base station, detection visualizer | Necessary | ✅ Complete |
| **Wk 6–7 — System Optimization:** Detection accuracy tuning, HSV fallback ranges, latency stabilization | Useful | ✅ Complete |
| **Wk 7–9 — Data-to-Action Pipeline:** Wireless link (router/USB-C), ground station logging, coordinate visualization | Useful | ✅ Complete |
| **Wk 9–10 — Integrated System Validation:** Full demo flight over buoys, field video recording, end-to-end demonstration | Useful | 🔄 In Progress |
| **Stretch — RC Car Integration:** RC car receives coordinate map, runs path planning | Hopeful | 🔲 Planned |
| **Stretch — Dynamic Monitoring Loop:** Live coordinate updates as buoys move | Hopeful | 🔲 Planned |
