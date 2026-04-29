# Driver Safety & Health Monitoring System

A multi-sensor driver safety system that monitors **driver physiology** (blood oxygen, heart rate, physical position), **environment** (atmospheric pressure), and **driver alertness** (eye gaze, drowsiness detection). The system combines an Arduino-based sensor hub with real-time Python visualization and a computer vision subsystem.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware Architecture](#hardware-architecture)
3. [Subsystems & Alarm Thresholds](#subsystems--alarm-thresholds)
   - [Atmospheric Pressure Sensor (LPS22HB)](#1-atmospheric-pressure-sensor-lps22hb)
   - [Proximity Sensor (APDS9960)](#2-proximity-sensor-apds9960)
   - [Oximeter / PPG (MAX86916)](#3-oximeter--ppg-max86916)
   - [Camera / Gaze Detection (iPhone)](#4-camera--gaze-detection-iphone)
4. [Software Architecture](#software-architecture)
5. [Setup & Usage](#setup--usage)
   - [Arduino: Flashing the Pressure/Oximeter Firmware](#arduino-flashing-the-pressureoximeter-firmware)
   - [Python: Running the Telemetry Dashboard](#python-running-the-telemetry-dashboard)
   - [Python: Running the Gaze Detection](#python-running-the-gaze-detection)
6. [Audio Files](#audio-files)
7. [Future Improvements](#future-improvements)

---

## System Overview

This project implements a **real-time driver monitoring system** that collects data from multiple sensors and displays them through a centralized Tkinter dashboard. The system has three tiers of alerts:

| Level | Description | Audio | Visual |
|-------|-------------|-------|--------|
| **0 — Stable** | All systems nominal | Silent | Green indicator |
| **1 — Warning** | Non-critical condition detected | `beep_short.wav` (looping) | Yellow background, black text |
| **2 — Critical** | Life-safety threshold breached | `alarm_loud.wav` (looping) | Red flashing background, white text |

An independent **computer vision subsystem** provides gaze tracking and drowsiness detection using an iPhone camera stream with MediaPipe Face Landmarks, displayed in its own OpenCV window with its own alert hierarchy.

---

## Hardware Architecture

The system uses a **single Arduino board** connected to the PC over USB serial, plus an iPhone for the computer vision component:

### Arduino Board — Sensor Hub (pressure + proximity + oximeter)

| Component | Part | Interface | Role |
|-----------|------|-----------|------|
| **Pressure** | LPS22HB (via `Arduino_LPS22HB`) | I²C | Measures atmospheric pressure |
| **Proximity** | APDS9960 (via `Arduino_APDS9960`) | I²C | Detects nearby objects |
| **Oximeter** | MAX86916 (via custom `MAX86916_eda` library) | I²C | Photoplethysmography (PPG) for SpO₂ & HR |
| **ADC/DAC** | AD5593R (via custom `AD5593R` library) | I²C | Multipurpose ADC/DAC (extensibility) |

### iPhone Camera (Computer Vision)

| Component | Details |
|-----------|---------|
| **Camera** | iPhone rear/front camera streamed to PC via IP Webcam or Continuity Camera |
| **Resolution** | HD+ (far superior to the OV7675 / OV7670 Arduino camera modules) |
| **Connection** | Wi-Fi (local network) or USB tethering |

The OV7675 / OV7670 camera modules were originally prototyped (`stream_test/Capture/Capture.ino`) but their low resolution and poor low-light performance made them unsuitable for reliable face landmark detection. The iPhone camera provides dramatically better image quality for MediaPipe-based gaze tracking.

---

## Subsystems & Alarm Thresholds

### 1. Atmospheric Pressure Sensor (LPS22HB)

Reads absolute atmospheric pressure in kilopascals (kPa).

| Condition | Threshold | Alert Level |
|-----------|-----------|-------------|
| Pressure low | `< 100.0 kPa` | **Critical (2)** |

- **Display:** `ATMOSPHERIC PRESSURE (kPa)` — numerical value with two decimal places.
- **Critical alert text:** `CRITICAL: LOW PRESSURE`
- Pressure is sampled each `loop()` cycle and sent in every serial packet.

### 2. Proximity Sensor (APDS9960)

Detects nearby objects using infrared proximity (returns a raw value). The sensor is polled each cycle; if no reading is available, it returns `-1`.

| Condition | Threshold | Alert Level |
|-----------|-----------|-------------|
| Proximity hazard | 3 consecutive readings > 250 | **Warning (1)** |

- **Display:** `PROXIMITY VECTOR` — windowed moving average (last 10 valid readings).
- **False-alarm protection:** if the sensor returns `-1` (not ready), that sample is silently skipped — the previous display value is retained, and the danger-detection buffer is not affected.
- The 3-consecutive-readings requirement prevents transient triggers.

### 3. Oximeter / PPG (MAX86916)

Uses red and infrared LEDs to perform photoplethysmography. The sensor sends raw 18-bit ADC values for both the **red** and **IR** channels.

| Condition | Threshold | Alert Level |
|-----------|-----------|-------------|
| Hypoxia (low SpO₂) | `0 < SpO₂ < 90%` | **Critical (2)** |
| Heart rate abnormal | `< 40 BPM` or `> 150 BPM` | **Critical (2)** |
| No finger detected | Red < 2000 **and** IR < 2000 | **Warning (1)** |

- **Display:**
  - `PPG RED CHANNEL` / `PPG IR CHANNEL` — raw ADC values.
  - `BLOOD OXYGEN (SpO2 %)` — computed from Ratio of Ratios: `SpO₂ = 110 − 25 × R`, displayed as windowed average (last 15 computations).
  - `HEART RATE (BPM)` — derived from FFT peak detection (0.5–3.0 Hz band) on a 500-sample FIFO buffer, displayed as windowed average (last 15 computations).
- **Processing:** The raw signal is bandpass-filtered (0.5–5 Hz) before FFT analysis to remove DC drift and high-frequency noise.
- **No-finger state:** triggers a **Warning (1)** with the message `WARNING: NO FINGER DETECTED`, plays `beep_short.wav`, and turns the background yellow.

### 4. Camera / Gaze Detection (iPhone)

This subsystem runs as a separate Python process (`stream_test/webcam_gaze.py`) using an iPhone camera as the video source. It uses **MediaPipe Face Landmarker** to detect 478 facial landmarks and compute:

- **Eye Aspect Ratio (EAR):** measures whether eyes are open or closed (threshold: EAR < 0.18 → "CLOSED").
- **Gaze direction:** horizontal (LEFT / CENTER / RIGHT) and vertical (UP / DOWN) based on iris position relative to the eye corners.

| Condition | Duration | Alert Level |
|-----------|----------|-------------|
| Eyes closed (drowsiness) | ≥ 1.2 seconds | **Critical (2)** — flashing red border, `alarm_loud.wav` |
| Gaze off-center (distraction) | ≥ 2.0 seconds | **Warning (1)** — flashing yellow border, `beep_short.wav` |

The iPhone camera replaces the original OV7675 / OV7670 Arduino camera modules which had insufficient resolution and poor low-light performance for reliable face landmark detection. The iPhone stream provides clean HD video over Wi-Fi or USB, dramatically improving gaze tracking accuracy.

---

## Software Architecture

```
project/
├── README.md                          ← This file
│
├── pressure/                          ← Arduino: Sensor Hub firmware
│   ├── pressure.ino                   ← Main loop (reads all sensors, serial out)
│   ├── MAX86916_eda.cpp / .h          ← MAX86916 PPG sensor driver
│   └── AD5593R.cpp / .h               ← AD5593R ADC/DAC driver
│
├── stream_test/
│   ├── Capture/
│   │   └── Capture.ino                ← Arduino: Camera firmware (deprecated — see note)
│   ├── webcam_gaze.py                 ← Python: MediaPipe gaze detection
│   └── face_landmarker.task           ← MediaPipe model file (auto-downloaded)
│
├── python/
│   └── main.py                        ← Python: Telemetry dashboard + signal processing
│
├── beep_short.wav                     ← Audio: Warning alert
└── alarm_loud.wav                     ← Audio: Critical alert
```

> **Note:** `Capture.ino` was the original Arduino camera firmware using OV7675 / OV7670. It has been superseded by the iPhone camera due to image quality limitations. The code is retained for reference but is no longer part of the active system.

The two subsystems (sensor dashboard and gaze detection) are designed to run **independently**. They can run on the same PC or on separate machines.

---

## Setup & Usage

### Arduino: Flashing the Pressure/Oximeter Firmware

1. Open `pressure/pressure.ino` in the Arduino IDE.
2. Install the required libraries:
   - `Arduino_LPS22HB`
   - `Arduino_APDS9960`
   - `Wire` (built-in)
3. Make sure `MAX86916_eda.cpp/.h` and `AD5593R.cpp/.h` are in the sketch folder.
4. Select the correct board and port (e.g., Arduino Nano 33 BLE, Port: `COM5`).
5. Click **Upload**.

### Python: Running the Telemetry Dashboard

1. Install required Python packages:

   ```bash
   pip install pyserial numpy scipy pygame
   ```

2. Connect the **sensor hub Arduino** via USB.
3. Run the dashboard:

   ```bash
   cd python
   python main.py
   ```

4. At the prompt, enter the serial port of the sensor hub Arduino:
   - Windows: `COM5`, `COM12`, etc.
   - Linux/WSL: `/dev/ttyUSB0`, `/dev/ttyACM0`, etc.

5. At the next prompt, enter the command to activate the PPG sensor:

   ```
   Type command (>): m3ppg
   ```

   > **Important:** The `m3ppg` command is sent to the Arduino to initialize the MAX86916 oximeter. You must type this exactly.

6. The dashboard window will open. Place your finger on the oximeter sensor to see SpO₂ and heart rate readings.

### Python: Running the Gaze Detection

1. Install additional packages:

   ```bash
   pip install opencv-python mediapipe numpy pygame
   ```

2. Set up your iPhone camera as a webcam source:
   - **Option A — Continuity Camera:** If on macOS, your iPhone can wirelessly act as a webcam (select it as camera index 0 or higher in OpenCV).
   - **Option B — IP Webcam:** Install an IP webcam app on your iPhone, then replace the camera source in `webcam_gaze.py` with the RTSP/MJPEG stream URL.

3. Run:

   ```bash
   cd stream_test
   python webcam_gaze.py
   ```

4. The face landmarker model (`face_landmarker.task`) is downloaded automatically on first run.
5. Press **Q** to quit.

---

## Audio Files

Two audio files are located in the project root:

| File | Purpose | Used By |
|------|---------|---------|
| `beep_short.wav` | Warning-level alert (looping) | `python/main.py` (no-finger, proximity), `stream_test/webcam_gaze.py` (distraction) |
| `alarm_loud.wav` | Critical-level alert (looping) | `python/main.py` (hypoxia, HR abnormal, low pressure), `stream_test/webcam_gaze.py` (drowsiness) |

---

## Future Improvements

### Steering Wheel Integration

The ultimate goal of this project is to **integrate all sensors into a steering wheel** for real-world driving safety. Planned improvements include:

- **Embedded sensor hub:** Mount the oximeter (MAX86916) into the steering wheel rim so the driver's finger rests naturally on the sensor while holding the wheel.
- **Pressure sensor:** Monitor cabin pressure changes (e.g., rapid altitude changes, airbag deployment).
- **Proximity sensor:** Detect hands-on-wheel vs. hands-off driving.
- **iPhone camera mount:** Mount the iPhone on the dashboard or steering column, pointed at the driver's face for gaze tracking and drowsiness detection.
- **Unified alert system:** Route all alerts through the vehicle's existing audio/visual infrastructure (dashboard indicators, speaker system).
- **Wireless communication:** Replace USB serial with Bluetooth or Wi-Fi for untethered operation.
- **Data logging:** Store trip data (heart rate, SpO₂, gaze events) for post-drive analysis.
- **Edge computing:** Run FFT and gaze detection on an on-board Raspberry Pi or Jetson Nano rather than a laptop.

---

*Project developed for USC EE105 — Embedded Systems*
