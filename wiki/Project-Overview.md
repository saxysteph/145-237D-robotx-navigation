# Project Overview

## Background

The [Maritime RobotX 2026 Challenge](https://robotx.org/) tasks international university teams with fielding three coordinated autonomous robots — an Uncrewed Aerial Vehicle (UAV), an Uncrewed Surface Vessel (USV), and an Uncrewed Underwater Vehicle (UUV) — that work together on disaster-response missions.

Our project contributes the **aerial intelligence and communication infrastructure**: the perception and data pipeline that allows the UAV to act as an overhead scout, identifying obstacles and transmitting actionable intelligence to the surface vessel.

---

## The Problem

Safe passage — one of RobotX's core missions — requires a vessel to navigate a gate course defined by colored buoys. A surface vessel operating at water level has limited visibility of the full course layout. A UAV flying overhead can see everything — but only if it can reliably detect, classify, and geolocate each buoy and transmit that map to the vessel in real time.

The 2024 RobotX competition revealed that aerial-to-surface communication was a critical gap across competing teams. This project directly addresses that gap.

---

## Approach

### Perception Stack (on Jetson Orin Nano)

Detection is a two-stage pipeline:

1. **YOLO** (`buoy_best.onnx`) — a lightweight YOLO11n model trained on buoy/balloon images, run via ONNX Runtime on the Jetson. Outputs bounding boxes around buoy candidates. Class labels from YOLO are intentionally ignored — only bounding box geometry is used.

2. **HSV Thresholding** — each bounding box ROI is cropped and classified in HSV color space. Hue ranges for red, green, and blue are applied to determine buoy color. This two-stage design (ML for localization, classical CV for color) is more robust than end-to-end classification on limited training data.

### Coordinate Mapping

Each detected buoy's pixel centroid is projected to GPS coordinates using:
- Drone altitude (meters)
- Camera heading (degrees)
- Camera field-of-view and intrinsic matrix
- Drone GPS position (lat/lon)

This flat-earth, nadir-camera projection model gives a real-world lat/lon estimate for each buoy at the cost of assuming a flat ground plane — valid for the competition environment.

### Communication

Detections are transmitted as **MAVLink STATUSTEXT** messages over UDP. Each packet encodes:
```
RXB|<target_id>|<color>|<lat_e7>|<lon_e7>|<frame>
```

The ground station (laptop or USV compute) listens on `udpin:0.0.0.0:14555`, decodes incoming packets, and logs detections as JSON.

Current link: **USB-C ethernet** (`192.168.55.x`). Planned: **USB WiFi dongle** for wireless operation.

---

## Minimum Viable Product

The MVP (demonstrated Week 5) proves the drone-to-receiver data pipeline end-to-end:

1. Drone is manually flown over simulated colored buoys (cones)
2. Jetson runs YOLO + HSV in real time, classifies each buoy
3. GPS coordinates are computed for each detection
4. Coordinates are transmitted via MAVLink to a base station laptop
5. Laptop logs and visualizes received detections as a GPS dot map

Physical actuation of the surface vehicle is a post-MVP stretch goal.

---

## Post-MVP Roadmap

The quarter's focus shifted to getting the full aerial pipeline working on the drone — the RC car surface integration is out of scope for this class deliverable. Future work for RobotX 2026 includes:

| Phase | Goal |
|---|---|
| Near-term | Wireless communication via portable router (in progress) |
| Near-term | Full demo flight over buoys with field video recording |
| Long-term | USV integration — USV subscribes to buoy coordinate map from UAV |
| Long-term | Autonomous UAV flight (search pattern, no manual pilot) |
| Long-term | Full tri-domain integration with UUV |

---

## Hardware

| Component | Details |
|---|---|
| Edge Computer | Jetson Orin Nano (JetPack 5, Python 3.8) |
| Camera | H264 USB camera |
| UAV Frame | Team Inspiration 2024 competition drone |
| Proxy USV | RC car (Prof. Silberman's lab) — post-scope, not implemented this quarter |
| Communication | USB-C ethernet (current), USB WiFi dongle (planned) |

## Software

| Component | Tool |
|---|---|
| Object detection | YOLO11n via Ultralytics + ONNX Runtime |
| Color classification | OpenCV HSV thresholding |
| Communication protocol | MAVLink (pymavlink) |
| Ground station | Custom Python listener |
| Visualization | matplotlib dot map |
| Language | Python 3.8+ |
