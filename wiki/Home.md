# RobotX Navigation — Aerial Intelligence & Communication Pipeline

> **CSE 145 / CSE 237D — Spring 2026 | UC San Diego**  
> Advised by Prof. Jack Silberman | In collaboration with TritonAI & Team Inspiration

---

## What This Project Does

A drone autonomously detects colored buoys from above, calculates their GPS coordinates, and transmits a real-time buoy map to a ground station — providing aerial intelligence for the [Maritime RobotX 2026 Challenge](https://robotx.org/).

> *"To successfully deploy an aerial-to-surface recognition system that extends the navigational intelligence of a surface vessel, the drone must integrate an edge-compute color thresholding model for obstacle detection, mathematical ray-projection logic for coordinate mapping, and a low-latency MAVLink publish-subscribe network for actionable payload transmission."*

---

## System Architecture

```
┌──────────────────────────────────────────┐
│               UAV (Drone)                │
│                                          │
│  H264 Camera                             │
│      ↓                                   │
│  YOLO11n (buoy_best.onnx)                │  ← bounding box proposals
│      ↓                                   │
│  HSV Thresholding                        │  ← color classification (R/G/B)
│      ↓                                   │
│  Ray Projection                          │  ← pixel → GPS lat/lon
│      ↓                                   │
│  MAVLink STATUSTEXT (UDP)                │
│                                          │
│  Jetson Orin Nano (JetPack 5)            │
└──────────────────┬───────────────────────┘
                   │  UDP · 14555
                   ▼
┌──────────────────────────────────────────┐
│          Ground Station (Laptop)         │
│                                          │
│  run_ground_station.py                   │  ← decodes & logs JSON
│  visualize_detections.py                 │  ← GPS dot map
└──────────────────────────────────────────┘
```

---

## Quick Links

| | Resource | Description |
|---|---|---|
| 🚀 | [Partner Run Instructions](https://github.com/saxysteph/145-237D-robotx-navigation/blob/main/fulldemo/PARTNER_INSTRUCTIONS.md) | **Start here to run the pipeline** |
| 📋 | [Full Demo Guide](https://github.com/saxysteph/145-237D-robotx-navigation/blob/main/fulldemo/README.md) | End-to-end demo steps with post-processing |
| 🔧 | [Setup & Replication](Setup-and-Replication) | Full dependency install and configuration |
| 📁 | [Repository Structure](Repository-Structure) | Codebase layout and key file descriptions |
| 🔬 | [Project Overview](Project-Overview) | Technical approach, MVP, and roadmap |
| 👥 | [Team](Team) | Team members and advisors |

---

## Project Status

| Milestone | Priority | Status |
|---|---|---|
| **Wk 4 — System Setup:** Jetson Orin setup, live camera feed, basic detection running | Necessary | ✅ Complete |
| **Wk 5 — MVP:** Buoy color detection, GPS coordinate projection, MAVLink transmission to base station, detection visualizer | Necessary | ✅ Complete |
| **Wk 6–7 — System Optimization:** Detection accuracy tuning, HSV fallback ranges, latency stabilization | Useful | ✅ Complete |
| **Wk 7–9 — Data-to-Action Pipeline:** Wireless link (router/USB-C), ground station logging, coordinate visualization | Useful | ✅ Complete |
| **Wk 9–10 — Integrated System Validation:** Full demo flight over buoys, field video recording, end-to-end demonstration | Useful | 🔄 In Progress |

---

## Repository

[`saxysteph/145-237D-robotx-navigation`](https://github.com/saxysteph/145-237D-robotx-navigation)
