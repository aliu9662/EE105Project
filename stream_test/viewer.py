"""
viewer.py — combined live preview and full capture viewer for the Arduino camera.

Preview mode:
    continuous 160x120 grayscale stream

Full mode:
    320x240 grayscale, one frame per SPACE press

Requirements:
    pip install pyserial opencv-python mediapipe numpy

Usage:
    python viewer.py
    python viewer.py --port /dev/cu.usbmodem21401 --baud 2000000 --mode preview

Keys:
    P        toggle preview/full mode
    SPACE    capture a frame in full mode
    R        cycle rotation
    M        toggle MediaPipe overlay
    Q        quit
"""

import argparse
import threading
import time
import pygame
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import serial
from serial.tools import list_ports

MAGIC = b"\xDE\xAD\xBE\xEF"
FMT_GRAY = 0x01

MODE_PREVIEW = "preview"
MODE_FULL = "full"

DEFAULT_BAUD = 2000000
WINDOW_NAME = "Arduino Gaze Viewer"
FULL_CAPTURE_TIMEOUT_S = 4.0

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

L_OUTER, L_INNER, L_TOP, L_BOT = 33, 133, 159, 145
R_OUTER, R_INNER, R_TOP, R_BOT = 263, 362, 386, 374
L_CTR, R_CTR = 468, 473

EYE_CLR = (255, 220, 0)
IRIS_CLR = (0, 200, 255)
TEXT_CLR = (0, 255, 0)
WARN_CLR = (0, 0, 255)
OPEN_CLR = (0, 220, 0)
CLOSED_CLR = (0, 0, 255)

ROTATIONS = [
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
]
ROTATION_LABELS = ["0°", "90°CW", "180°", "90°CCW"]


class CVAlertManager:
    def __init__(self):
        pygame.mixer.init()
        self.snd_distract = pygame.mixer.Sound('beep_short.mp3')
        self.snd_fatigue = pygame.mixer.Sound('alarm_loud.mp3')
        
        self.eyes_closed_since = None
        self.distracted_since = None
        self.DROWSY_THRESHOLD = 1.5 
        self.DISTRACT_THRESHOLD = 2.0
        
        self.alert_level = 0
        self.flash_state = False
        self.last_flash_time = time.monotonic()

    def update_state(self, eyes_state, gaze_direction):
        now = time.monotonic()
        
        # 1. Eyes Closed Logic
        if eyes_state == "CLOSED":
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
        else:
            self.eyes_closed_since = None
            
        # 2. Distraction Logic
        if gaze_direction and "CENTER" not in gaze_direction:
            if self.distracted_since is None:
                self.distracted_since = now
        else:
            self.distracted_since = None
            
        # 3. Evaluate Alerts
        if self.eyes_closed_since and (now - self.eyes_closed_since) >= self.DROWSY_THRESHOLD:
            self._trigger(2)
        elif self.distracted_since and (now - self.distracted_since) >= self.DISTRACT_THRESHOLD:
            self._trigger(1)
        else:
            self._trigger(0)

    def _trigger(self, level):
        if self.alert_level != level:
            self.alert_level = level
            pygame.mixer.stop()
            # if level == 1: pygame.mixer.Sound.play(self.snd_distract)
            # if level == 2: pygame.mixer.Sound.play(self.snd_fatigue, loops=-1)

    def draw_alert(self, frame):
        """Draws a flashing border and text directly onto the OpenCV frame."""
        if self.alert_level == 0:
            return frame
            
        now = time.monotonic()
        if now - self.last_flash_time > 0.4:
            self.flash_state = not self.flash_state
            self.last_flash_time = now
            
        h, w = frame.shape[:2]
        
        if self.alert_level == 1:
            color = (0, 255, 255) if self.flash_state else (0, 150, 150) # Yellow (BGR)
            text = "WARNING: EYES ON ROAD"
        elif self.alert_level == 2:
            color = (0, 0, 255) if self.flash_state else (0, 0, 100) # Red (BGR)
            text = "CRITICAL: WAKE UP!"
            
        # Draw border
        cv2.rectangle(frame, (0, 0), (w, h), color, 15)
        # Draw text background
        cv2.rectangle(frame, (w//2 - 200, 20), (w//2 + 200, 80), (0,0,0), -1)
        # Draw text
        cv2.putText(frame, text, (w//2 - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return frame


def load_detector():
    if not MODEL_PATH.exists():
        print(f"Downloading face landmarker -> {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("Done.")
    opts = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _mp_vision.FaceLandmarker.create_from_options(opts)


def gaze_direction(lm):
    def x(i):
        return lm[i].x

    def y(i):
        return lm[i].y

    h = (
        (x(L_CTR) - x(L_OUTER)) / max(x(L_INNER) - x(L_OUTER), 1e-6)
        + (x(R_CTR) - x(R_INNER)) / max(x(R_OUTER) - x(R_INNER), 1e-6)
    ) / 2.0
    v = (
        (y(L_CTR) - y(L_TOP)) / max(y(L_BOT) - y(L_TOP), 1e-6)
        + (y(R_CTR) - y(R_TOP)) / max(y(R_BOT) - y(R_TOP), 1e-6)
    ) / 2.0

    h_label = "LEFT" if h < 0.38 else "RIGHT" if h > 0.62 else "CENTER"
    v_label = " UP" if v < 0.35 else " DOWN" if v > 0.65 else ""
    return h_label + v_label, round(h, 3), round(v, 3)


def lm_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)


def draw_gaze_overlay(frame, lm):
    h, w = frame.shape[:2]
    for idx in LEFT_EYE + RIGHT_EYE:
        cv2.circle(frame, lm_px(lm[idx], w, h), 2, EYE_CLR, -1)
    for iris_ids, ctr in ((LEFT_IRIS, L_CTR), (RIGHT_IRIS, R_CTR)):
        pts = [lm_px(lm[i], w, h) for i in iris_ids]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = lm_px(lm[ctr], w, h)
        cv2.ellipse(
            frame,
            (cx, cy),
            (max(1, (max(xs) - min(xs)) // 2), max(1, (max(ys) - min(ys)) // 2)),
            0,
            0,
            360,
            IRIS_CLR,
            2,
        )
        cv2.circle(frame, (cx, cy), 2, IRIS_CLR, -1)


class PreviewEyeTracker:
    def __init__(self):
        self._prev_centers = None

    def reset(self):
        self._prev_centers = None

    def _eye_label(self, rect, darkness):
        _, _, w, h = rect
        openness = h / max(w, 1)
        closed = openness < 0.38 and darkness < 0.28
        return {
            "openness": round(openness, 3),
            "darkness": round(darkness, 3),
            "label": "CLOSED" if closed else "OPEN",
            "color": CLOSED_CLR if closed else OPEN_CLR,
        }

    def _candidates(self, gray):
        h, w = gray.shape[:2]
        y1 = int(h * 0.12)
        y2 = max(y1 + 1, int(h * 0.78))
        x1 = int(w * 0.18)
        x2 = max(x1 + 1, int(w * 0.82))
        roi = gray[y1:y2, x1:x2]

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        thresh_val = int(min(110, np.percentile(blur, 40)))
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        roi_area = roi.shape[0] * roi.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8 or area > roi_area * 0.06:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 4 or ch < 3:
                continue

            aspect = cw / max(ch, 1)
            fill = area / max(cw * ch, 1)
            if aspect < 0.35 or aspect > 4.0 or fill < 0.12:
                continue

            roi_values = roi[y:y + ch, x:x + cw]
            darkness = float(1.0 - np.mean(roi_values) / 255.0)
            if darkness < 0.18:
                continue
            center = (x1 + x + cw / 2.0, y1 + y + ch / 2.0)
            candidates.append(
                {
                    "rect": (x1 + x, y1 + y, cw, ch),
                    "center": center,
                    "area": area,
                    "aspect": aspect,
                    "darkness": darkness,
                }
            )
        return candidates

    def _pick_pair(self, candidates, frame_w, frame_h):
        best = None
        best_score = -1e9

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                left = candidates[i]
                right = candidates[j]
                if left["center"][0] > right["center"][0]:
                    left, right = right, left

                dx = right["center"][0] - left["center"][0]
                dy = abs(right["center"][1] - left["center"][1])
                if dx < frame_w * 0.08 or dx > frame_w * 0.38:
                    continue
                if dy > frame_h * 0.16:
                    continue

                lw = left["rect"][2]
                rw = right["rect"][2]
                lh = left["rect"][3]
                rh = right["rect"][3]
                size_penalty = abs(lw - rw) + abs(lh - rh)
                score = (left["darkness"] + right["darkness"]) * 8.0
                score += (left["aspect"] + right["aspect"]) * 0.8
                score -= dy * 0.08
                score -= size_penalty * 0.1
                score -= abs((left["center"][1] + right["center"][1]) * 0.5 - frame_h * 0.42) * 0.05

                if self._prev_centers is not None:
                    score -= (
                        abs(left["center"][0] - self._prev_centers[0][0])
                        + abs(left["center"][1] - self._prev_centers[0][1])
                        + abs(right["center"][0] - self._prev_centers[1][0])
                        + abs(right["center"][1] - self._prev_centers[1][1])
                    ) * 0.06

                if score > best_score:
                    best_score = score
                    best = (left, right)

        return best

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = self._candidates(gray)
        pair = self._pick_pair(candidates, gray.shape[1], gray.shape[0])
        if pair is None:
            self._prev_centers = None
            return None

        left, right = pair
        self._prev_centers = (left["center"], right["center"])
        left.update(self._eye_label(left["rect"], left["darkness"]))
        right.update(self._eye_label(right["rect"], right["darkness"]))
        eyes_label = "CLOSED" if left["label"] == "CLOSED" and right["label"] == "CLOSED" else "OPEN"
        return {"left": left, "right": right, "eyes": eyes_label}


def draw_eye_state_overlay(frame, left_eye, right_eye):
    for eye in (left_eye, right_eye):
        x, y, w, h = eye["rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), eye["color"], 2)
        cv2.circle(frame, (int(eye["center"][0]), int(eye["center"][1])), 2, eye["color"], -1)
        cv2.putText(
            frame,
            eye["label"],
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            eye["color"],
            1,
        )


def pick_port(requested):
    if requested:
        return requested
    ports = list_ports.comports()
    candidates = [
        p
        for p in ports
        if "usbmodem" in p.device or "Nano" in p.description or "Arduino" in p.description
    ] or ports
    if not candidates:
        raise RuntimeError("No serial ports found.")
    if len(candidates) == 1:
        print(f"Auto-selected: {candidates[0].device}")
        return candidates[0].device
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p.device}  -  {p.description}")
    return candidates[int(input("Select port: "))].device


def make_placeholder(mode, message):
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    subtitle = "Live preview 160x120" if mode == MODE_PREVIEW else "On-demand full capture 320x240"
    cv2.putText(canvas, subtitle, (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
    cv2.putText(canvas, message, (110, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    return canvas


def fit_for_display(frame, max_w=960, max_h=720):
    h, w = frame.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    scale = max(scale, 1.0)
    interp = cv2.INTER_NEAREST if w <= 160 else cv2.INTER_LINEAR
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=interp)


def render_frame(raw_frame, info, rot_idx, show_mp, detector, mode, preview_tracker):
    canvas = raw_frame.copy()
    if ROTATIONS[rot_idx] is not None:
        canvas = cv2.rotate(canvas, ROTATIONS[rot_idx])

    status = {
        "detected": False,
        "mode": mode,
        "width": info["width"],
        "height": info["height"],
    }

    if mode == MODE_PREVIEW:
        found = preview_tracker.detect(canvas)
        if found:
            status["detected"] = True
            left_eye = found["left"]
            right_eye = found["right"]
            draw_eye_state_overlay(canvas, left_eye, right_eye)
            eyes_color = CLOSED_CLR if found["eyes"] == "CLOSED" else OPEN_CLR
            cv2.putText(canvas, f"Eyes: {found['eyes']}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eyes_color, 2)
            cv2.putText(
                canvas,
                f"L o={left_eye['openness']:.2f} d={left_eye['darkness']:.2f}   "
                f"R o={right_eye['openness']:.2f} d={right_eye['darkness']:.2f}",
                (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                eyes_color,
                1,
            )
            status.update(found)
        else:
            cv2.putText(canvas, "Eyes not found", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WARN_CLR, 2)
    elif show_mp:
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            status["detected"] = True
            gaze, h_ratio, v_ratio = gaze_direction(lm)
            draw_gaze_overlay(canvas, lm)
            cv2.putText(canvas, f"Gaze: {gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_CLR, 2)
            cv2.putText(
                canvas,
                f"h={h_ratio:.2f}  v={v_ratio:.2f}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                TEXT_CLR,
                1,
            )
            status.update({"gaze": gaze, "h": h_ratio, "v": v_ratio})
        else:
            cv2.putText(canvas, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WARN_CLR, 2)

    return fit_for_display(canvas), status


class SerialReceiver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ser = serial.Serial()
        self._running = True
        self._buf = bytearray()
        self._text = bytearray()
        self._state = "search"
        self._frame_header = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_info = None
        self._seq = 0

    def connect(self, port, baud):
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.05
        self.ser.open()
        print(f"Connected: {port} @ {baud}")

    def send(self, cmd):
        if self.ser.is_open:
            self.ser.write((cmd + "\n").encode("utf-8"))
            print(f"-> {cmd}")

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._text.clear()
            self._state = "search"
            self._frame_header = None
        if self.ser.is_open:
            self.ser.reset_input_buffer()

    def latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None, None, None
            return self._seq, self._frame.copy(), dict(self._frame_info)

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        if self.ser.is_open:
            self.ser.close()

    def run(self):
        while self._running:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                self._buf.extend(chunk)
                self._process()
            except serial.SerialException as exc:
                print(f"Serial error: {exc}")
                time.sleep(0.1)
            except Exception as exc:
                print(f"Receiver error: {exc}")
                time.sleep(0.1)

    def _flush_text(self, data, flush_partial=False):
        if not data:
            return
        self._text.extend(data)

        while True:
            positions = [idx for idx in (self._text.find(b"\n"), self._text.find(b"\r")) if idx != -1]
            if not positions:
                break
            idx = min(positions)
            line = self._text[:idx].decode("utf-8", errors="replace").strip()
            del self._text[: idx + 1]
            if line:
                print(f"[arduino] {line}")

        if flush_partial and self._text:
            line = self._text.decode("utf-8", errors="replace").strip()
            self._text.clear()
            if line:
                print(f"[arduino] {line}")

    def _process(self):
        while True:
            if self._state == "search":
                idx = self._buf.find(MAGIC)
                if idx == -1:
                    keep = len(MAGIC) - 1
                    safe = max(0, len(self._buf) - keep)
                    if safe:
                        self._flush_text(bytes(self._buf[:safe]), flush_partial=False)
                        del self._buf[:safe]
                    return

                if idx > 0:
                    self._flush_text(bytes(self._buf[:idx]), flush_partial=True)
                    del self._buf[:idx]

                del self._buf[:4]
                self._state = "header"

            elif self._state == "header":
                if len(self._buf) < 9:
                    return
                fmt = self._buf[0]
                width = int.from_bytes(self._buf[1:3], "little")
                height = int.from_bytes(self._buf[3:5], "little")
                size = int.from_bytes(self._buf[5:9], "little")
                del self._buf[:9]

                if fmt != FMT_GRAY:
                    print(f"Unknown format byte: {fmt:#x}")
                    self._state = "search"
                    continue

                expected = width * height
                if width <= 0 or height <= 0 or size != expected:
                    print(f"Bad frame header: {width}x{height} size={size} expected={expected}")
                    self._state = "search"
                    continue

                self._frame_header = (width, height, size)
                self._state = "payload"

            else:
                width, height, size = self._frame_header
                if len(self._buf) < size:
                    return

                pixels = bytes(self._buf[:size])
                del self._buf[:size]
                arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width))
                frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

                with self._lock:
                    self._seq += 1
                    self._frame = frame
                    self._frame_info = {"width": width, "height": height}

                self._state = "search"
                self._frame_header = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--rotate", type=int, default=90, choices=[0, 90, 180, 270])
    parser.add_argument("--mode", default=MODE_PREVIEW, choices=[MODE_PREVIEW, MODE_FULL])
    args = parser.parse_args()

    print("Loading MediaPipe for full mode...")
    detector = load_detector()
    print("Ready.")

    port = pick_port(args.port)
    receiver = SerialReceiver()
    receiver.connect(port, args.baud)
    time.sleep(1.5)
    receiver.ser.reset_input_buffer()
    receiver.start()

    mode = args.mode
    rot_idx = [0, 90, 180, 270].index(args.rotate)
    show_mp = True
    preview_tracker = PreviewEyeTracker()

    raw_frame = None
    rendered = None
    last_status = None
    last_seq = -1
    dirty = True
    last_frame_at = time.monotonic()
    last_preview_request_at = 0.0
    capture_pending = False
    capture_requested_at = 0.0
    status_line = ""

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    receiver.send("STATUS")

    if mode == MODE_FULL:
        receiver.clear()
        receiver.send("MODE FULL")
        status_line = "Full mode armed. Press SPACE to capture."

    print("Keys: P=toggle mode  SPACE=capture(full only)  R=rotate  M=mediapipe  Q=quit")

    alert_manager = CVAlertManager()

    while True:
        seq, frame, info = receiver.latest_frame()
        if seq is not None and seq != last_seq:
            last_seq = seq
            raw_frame = frame
            dirty = True
            last_frame_at = time.monotonic()
            if capture_pending and mode == MODE_FULL:
                capture_pending = False
                status_line = f"Full frame received: {info['width']}x{info['height']}"

        if raw_frame is not None and dirty:
            rendered, last_status = render_frame(raw_frame, info, rot_idx, show_mp, detector, mode, preview_tracker)
            dirty = False

            if last_status["detected"] and mode == MODE_PREVIEW:
                print(
                    f"Eyes: {last_status['eyes']:<6} "
                    f"L(o={last_status['left']['openness']:.3f}, d={last_status['left']['darkness']:.3f}) "
                    f"R(o={last_status['right']['openness']:.3f}, d={last_status['right']['darkness']:.3f}) "
                    f"frame={last_status['width']}x{last_status['height']}"
                )
            elif last_status["detected"]:
                print(
                    f"Gaze: {last_status['gaze']:<16} "
                    f"h={last_status['h']:.3f}  v={last_status['v']:.3f}  "
                    f"frame={last_status['width']}x{last_status['height']}"
                )
            else:
                if mode == MODE_PREVIEW:
                    print(f"Eyes not found  frame={last_status['width']}x{last_status['height']}")
                else:
                    print(f"No face detected  frame={last_status['width']}x{last_status['height']}")
            current_eyes = "OPEN"
            current_gaze = "CENTER"

            if last_status["detected"]:
                if mode == MODE_PREVIEW:
                    # PREVIEW mode tracks open/closed eyes via OpenCV contours
                    current_eyes = last_status.get("eyes", "OPEN")
                else:
                    # FULL mode tracks gaze direction via MediaPipe
                    current_gaze = last_status.get("gaze", "CENTER")
            else:
                # STRICT SAFETY PROTOCOL: If no face/eyes are found, assume distracted.
                # Passing "MISSING" (which doesn't contain "CENTER") starts the 2.0-second distraction timer.
                current_gaze = "MISSING"
                
            alert_manager.update_state(current_eyes, current_gaze)

        if rendered is None:
            prompt = "Waiting for preview..." if mode == MODE_PREVIEW else "Press SPACE to capture"
            canvas = make_placeholder(mode, prompt)
        else:
            canvas = rendered.copy()

        now = time.monotonic()
        if mode == MODE_PREVIEW and now - last_frame_at > 2.0 and now - last_preview_request_at > 2.0:
            receiver.send("MODE PREVIEW")
            last_preview_request_at = now
        if capture_pending and now - capture_requested_at > FULL_CAPTURE_TIMEOUT_S:
            capture_pending = False
            status_line = "Full capture timed out."
            print(status_line)

        mode_label = "PREVIEW" if mode == MODE_PREVIEW else "FULL"
        overlay_label = "CV" if mode == MODE_PREVIEW else ("MP:on" if show_mp else "MP:off")
        rot_label = ROTATION_LABELS[rot_idx]
        cv2.putText(
            canvas,
            f"Mode:{mode_label}  R:{rot_label}  {overlay_label}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        if status_line:
            cv2.putText(
                canvas,
                status_line,
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
        cv2.putText(
            canvas,
            "P=toggle  SPACE=full capture  R=rotate  M=overlay  Q=quit",
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (140, 140, 140),
            1,
        )

        canvas = alert_manager.draw_alert(canvas)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(15) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            rot_idx = (rot_idx + 1) % len(ROTATIONS)
            dirty = raw_frame is not None
        elif key == ord("m"):
            show_mp = not show_mp
            dirty = raw_frame is not None
            print(f"MediaPipe: {'on' if show_mp else 'off'}")
        elif key == ord("p"):
            if mode == MODE_PREVIEW:
                mode = MODE_FULL
                raw_frame = None
                rendered = None
                last_status = None
                dirty = False
                last_frame_at = time.monotonic()
                capture_pending = False
                preview_tracker.reset()
                receiver.clear()
                receiver.send("MODE FULL")
                status_line = "Full mode armed. Press SPACE to capture."
                print("Mode: FULL")
            else:
                mode = MODE_PREVIEW
                raw_frame = None
                rendered = None
                last_status = None
                dirty = False
                last_frame_at = time.monotonic()
                capture_pending = False
                preview_tracker.reset()
                receiver.clear()
                receiver.send("MODE PREVIEW")
                last_preview_request_at = time.monotonic()
                status_line = "Preview mode."
                print("Mode: PREVIEW")
        elif key == ord(" "):
            if mode == MODE_FULL:
                receiver.clear()
                receiver.send("CAPTURE")
                capture_pending = True
                capture_requested_at = time.monotonic()
                status_line = "Waiting for full capture..."
                print("Capture requested")

    receiver.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
