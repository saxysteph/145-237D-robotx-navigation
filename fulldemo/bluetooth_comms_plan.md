# Bluetooth Communication Plan — Jetson Orin Nano ↔ Laptop

> **Status: PLANNED — do not implement until USB-C pipeline is verified end-to-end.**

## Hardware

- Jetson Orin Nano (no WiFi board)
- TP-Link UB400 nano USB Bluetooth 4.0 dongle (plugged into Jetson)
- MacBook Pro (built-in Bluetooth)

---

## Overview

Replace the USB-C ethernet link (`192.168.55.x`) with a Bluetooth PAN (Personal Area Network).
Bluetooth PAN creates a standard IP network over BT — the MAVLink UDP packets require no changes,
only the IP addresses and network interface names change.

Architecture stays identical:
```
Jetson (BT PAN client) → udpout:<laptop-BT-IP>:14555 → Laptop (BT PAN server, GCS)
```

---

## Step 1 — Verify UB400 is Recognized on Jetson

```bash
lsusb | grep -i "TP-Link\|Bluetooth"
hciconfig -a
```

Expected: `hci0` appears with the UB400's MAC address.
If not: `sudo apt-get install -y bluez bluetooth` and recheck.

---

## Step 2 — Pair Jetson and Laptop

On the Jetson:
```bash
sudo bluetoothctl
power on
agent on
scan on
# Note the MacBook's Bluetooth MAC (XX:XX:XX:XX:XX:XX)
pair XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
quit
```

On the Mac: accept the pairing request in System Settings → Bluetooth.

---

## Step 3 — Set Up Bluetooth PAN

Bluetooth PAN (profile: NAP — Network Access Point) lets one device act as a network gateway.

**Option A: Laptop as NAP server (recommended)**

The Mac shares a network over Bluetooth. The Jetson connects as a client and gets an IP.

On the Mac:
- System Settings → General → Sharing → **Internet Sharing**
- Share from: Wi-Fi (or Ethernet), To: **Bluetooth PAN**
- Enable Internet Sharing

The Mac will create a `bridge` or `btpan0` interface, typically in the `192.168.X.X` range.
The Jetson will get an IP via DHCP on that subnet.

On the Jetson:
```bash
sudo bt-network -c XX:XX:XX:XX:XX:XX nap   # XX = MacBook BT MAC
dhclient bnep0
ip addr show bnep0   # note the assigned IP
```

**Option B: Jetson as NAP server**

More complex since the Jetson has no internet to share. Not recommended for this use case.

---

## Step 4 — Determine IP Addresses

After PAN is up:

On the Jetson:
```bash
ip addr show bnep0
# e.g. 192.168.X.Y — this is the Jetson's BT IP
```

On the Mac:
```bash
ifconfig | grep -A2 bridge   # or btpan0
# e.g. 192.168.X.1 — this is the laptop's BT IP (GCS address)
```

---

## Step 5 — Update Pipeline for Bluetooth IP

No code changes needed — just pass the new laptop BT IP to `--gcs-ip`:

**Jetson:**
```bash
python3 camera_live_feed.py \
  --headless \
  --save-video \
  --camera-index 0 \
  --yolo-model buoy_best.pt \
  --gcs-ip <LAPTOP_BT_IP> \
  --drone-lat <LAT> --drone-lon <LON>
```

**Laptop GCS** (no change needed — listens on `0.0.0.0:14555`):
```bash
python mavlink_comms/scripts/run_ground_station.py
```

---

## Step 6 — Auto-reconnect on Boot (optional, post-demo)

To make the Jetson reconnect BT PAN automatically on power-up:

```bash
# /etc/systemd/system/bt-pan.service
[Unit]
Description=Bluetooth PAN connection to GCS
After=bluetooth.target

[Service]
ExecStart=/usr/bin/bt-network -c XX:XX:XX:XX:XX:XX nap
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable bt-pan
sudo systemctl start bt-pan
```

---

## Expected Performance

| Metric | USB-C Ethernet | Bluetooth 4.0 PAN |
|---|---|---|
| Bandwidth | ~1 Gbps | ~2 Mbps |
| Latency | <1ms | 10–50ms |
| Range | 0m (tethered) | ~10m (class 2) |
| MAVLink STATUSTEXT size | ~50 bytes | trivial |

MAVLink `STATUSTEXT` messages are ~50 bytes each — Bluetooth bandwidth is not a bottleneck.
Added latency (~10-50ms) is negligible for buoy GPS reporting.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| UB400 not recognized on Jetson kernel | Check `dmesg` for USB errors; may need `btusb` module: `sudo modprobe btusb` |
| BT PAN DHCP gives different IP each session | Assign static IP via `/etc/dhcpcd.conf` or use mDNS hostname |
| Mac BT PAN sharing not stable | Use direct pairing + `bt-network` instead of Internet Sharing |
| Range insufficient on drone | UB400 is class 2 (~10m); for longer range use class 1 dongle |

---

## Implementation Order (when ready)

1. Verify USB-C pipeline end-to-end ✓ (current)
2. Plug UB400 into Jetson, verify `hci0` appears
3. Pair devices
4. Bring up PAN, confirm `bnep0` gets IP
5. Run pipeline with `--gcs-ip <BT_IP>`, confirm `[GCS]` output on laptop
6. Test range: walk laptop away from Jetson, verify no packet loss
7. Add systemd auto-reconnect for field deployment
