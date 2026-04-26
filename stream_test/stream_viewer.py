"""
stream_viewer.py — live camera stream with MediaPipe iris / gaze overlay.

Requirements:
    pip install pyserial Pillow mediapipe numpy opencv-python

Usage:
    python stream_viewer.py
    python stream_viewer.py --port /dev/cu.usbmodem21401 --baud 2000000
"""

import argparse
import base64
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

import pygame
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
from pathlib import Path
import urllib.request
from PIL import Image, ImageDraw, ImageTk
import serial
from serial.tools import list_ports

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── EIML framing (must match Arduino sketch) ──────────────────────────────
EIML_SOF_B64   = b'/6D/'
EIML_SOF_SIZE  = 3
EIML_GRAYSCALE = 1
EIML_RGB888    = 2

DEFAULT_BAUD   = 2000000
TICK_MS        = 16     # ~60 Hz GUI refresh

# ── MediaPipe model (auto-downloaded on first run) ─────────────────────────
_MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'
_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
               'face_landmarker/face_landmarker/float16/1/face_landmarker.task')

def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        print(f'Downloading face landmarker model → {_MODEL_PATH} ...')
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print('Download complete.')
    return str(_MODEL_PATH)

def _init_detector():
    opts = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=_ensure_model()),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _mp_vision.FaceLandmarker.create_from_options(opts)

_detector = _init_detector()

# ── MediaPipe landmark indices ─────────────────────────────────────────────
# fmt: off
_LEFT_EYE_CONTOUR  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE_CONTOUR = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_IRIS         = [468,469,470,471,472]
_RIGHT_IRIS        = [473,474,475,476,477]

_L_OUTER, _L_INNER, _L_TOP, _L_BOT = 33,  133, 159, 145
_R_OUTER, _R_INNER, _R_TOP, _R_BOT = 263, 362, 386, 374
_L_CTR, _R_CTR                      = 468, 473
# fmt: on

C_EYE  = (0,   220, 255)
C_IRIS = (255, 200,   0)


# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------

def _gaze_ratios(lm):
    def x(i): return lm[i].x
    def y(i): return lm[i].y

    h_l = (x(_L_CTR) - x(_L_OUTER)) / max(x(_L_INNER) - x(_L_OUTER), 1e-6)
    h_r = (x(_R_CTR) - x(_R_INNER)) / max(x(_R_OUTER) - x(_R_INNER), 1e-6)
    h   = (h_l + h_r) / 2.0

    v_l = (y(_L_CTR) - y(_L_TOP)) / max(y(_L_BOT) - y(_L_TOP), 1e-6)
    v_r = (y(_R_CTR) - y(_R_TOP)) / max(y(_R_BOT) - y(_R_TOP), 1e-6)
    v   = (v_l + v_r) / 2.0

    h_label = 'LEFT'   if h < 0.38 else 'RIGHT' if h > 0.62 else 'CENTER'
    v_label = ' UP'    if v < 0.35 else ' DOWN'  if v > 0.65 else ''
    return h_label + v_label, round(h, 3), round(v, 3)


def _px(lm_pt, w, h):
    return (int(lm_pt.x * w), int(lm_pt.y * h))


def annotate(img: Image.Image):
    """
    Run FaceLandmarker and draw eye/iris overlay.
    Returns (annotated_RGB_image, info_dict).
    info_dict keys: detected, gaze, h_ratio, v_ratio, num_landmarks
    """
    rgb    = np.array(img.convert('RGB'))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_img)

    out = img.convert('RGB')

    if not result.face_landmarks:
        return out, {'detected': False, 'gaze': None,
                     'h_ratio': None, 'v_ratio': None, 'num_landmarks': 0}

    lm   = result.face_landmarks[0]
    w, h = img.size
    draw = ImageDraw.Draw(out)
    r    = max(1, w // 80)

    for idx in _LEFT_EYE_CONTOUR + _RIGHT_EYE_CONTOUR:
        cx, cy = _px(lm[idx], w, h)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=C_EYE)

    for iris_ids, ctr_id in ((_LEFT_IRIS, _L_CTR), (_RIGHT_IRIS, _R_CTR)):
        pts = [_px(lm[i], w, h) for i in iris_ids]
        xs, ys = zip(*pts)
        cx, cy = _px(lm[ctr_id], w, h)
        rx = max(1, (max(xs) - min(xs)) // 2)
        ry = max(1, (max(ys) - min(ys)) // 2)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=C_IRIS, width=max(1, r))
        draw.ellipse((cx - r,  cy - r,  cx + r,  cy + r),  fill=C_IRIS)

    gaze, h_ratio, v_ratio = _gaze_ratios(lm)

    def dist(p1, p2): return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    
    # Use your existing landmark constants to measure eyelid distance
    ear_l = dist(lm[_L_TOP], lm[_L_BOT]) / max(dist(lm[_L_OUTER], lm[_L_INNER]), 1e-6)
    ear_r = dist(lm[_R_TOP], lm[_R_BOT]) / max(dist(lm[_R_OUTER], lm[_R_INNER]), 1e-6)
    
    # Average the two eyes. If EAR < 0.15, the eyes are considered closed.
    avg_ear = (ear_l + ear_r) / 2.0
    eyes_state = "CLOSED" if avg_ear < 0.15 else "OPEN"
    
    return out, {'detected': True, 'gaze': gaze,
                 'h_ratio': h_ratio, 'v_ratio': v_ratio,
                 'num_landmarks': len(lm)}


# ---------------------------------------------------------------------------
# Serial receiver thread
# ---------------------------------------------------------------------------

class Receiver(threading.Thread):
    def __init__(self, on_frame, on_log):
        super().__init__(daemon=True)
        self.on_frame  = on_frame
        self.on_log    = on_log
        self.ser       = serial.Serial()
        self._running  = True
        self._frame_n  = 0

    def connect(self, port, baud):
        if self.ser.is_open:
            self.ser.close()
        self.ser.port     = port
        self.ser.baudrate = baud
        self.ser.timeout  = 0.1
        self.ser.open()
        self.on_log(f'Serial opened: {port} @ {baud}')

    def send(self, cmd: str):
        if self.ser.is_open:
            self.ser.write((cmd + '\n').encode())
            self.on_log(f'→ Sent command: {cmd}')

    def run(self):
        buf = b''
        while self._running:
            try:
                n = self.ser.in_waiting
                if n:
                    buf += self.ser.read(n)
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        line = line.rstrip(b'\r')
                        if line.startswith(EIML_SOF_B64):
                            self._decode(line)
                        elif line:
                            self.on_log(f'[arduino] {line.decode("utf-8", errors="replace")}')
                else:
                    time.sleep(0.002)
            except serial.SerialException as e:
                self.on_log(f'Serial error: {e}')
                time.sleep(0.05)
            except Exception as e:
                self.on_log(f'Receiver error: {e}')
                time.sleep(0.05)

    def _decode(self, line):
        try:
            raw = base64.b64decode(line)
            idx = EIML_SOF_SIZE
            fmt = raw[idx];             idx += 1
            w   = int.from_bytes(raw[idx:idx+4], 'little'); idx += 4
            h   = int.from_bytes(raw[idx:idx+4], 'little'); idx += 4
            px  = raw[idx:]
            img = (Image.frombytes('L', (w, h), px) if fmt == EIML_GRAYSCALE
                   else Image.frombytes('RGB', (w, h), px))
            self._frame_n += 1
            self.on_log(f'Frame #{self._frame_n}: {w}×{h}, {len(px)} bytes')
            self.on_frame(img)
        except Exception as e:
            self.on_log(f'Decode error: {e}')


# ---------------------------------------------------------------------------
# Webcam capture thread
# ---------------------------------------------------------------------------

class WebcamCapture(threading.Thread):
    def __init__(self, on_frame, on_log):
        super().__init__(daemon=True)
        self.on_frame  = on_frame
        self.on_log    = on_log
        self._running  = True
        self._active   = False
        self._cap      = None
        self._lock     = threading.Lock()

    def start_capture(self):
        if not CV2_AVAILABLE:
            self.on_log('ERROR: opencv-python not installed — run: pip install opencv-python')
            return
        with self._lock:
            if self._cap is None:
                self._cap = cv2.VideoCapture(0)
                if self._cap.isOpened():
                    self._active = True
                    self.on_log('Webcam opened')
                else:
                    self._cap = None
                    self.on_log('ERROR: could not open webcam')

    def stop_capture(self):
        with self._lock:
            self._active = False
            if self._cap:
                self._cap.release()
                self._cap = None
        self.on_log('Webcam closed')

    def run(self):
        while self._running:
            with self._lock:
                active = self._active
                cap    = self._cap
            if active and cap:
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.on_frame(Image.fromarray(rgb))
                time.sleep(1 / 30)
            else:
                time.sleep(0.05)

# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------

class DriverAlertManager:
    def __init__(self, root_window):
        self.root = root_window
        
        # Initialize Audio Mixer (Non-blocking)
        pygame.mixer.init()
        self.snd_distract = pygame.mixer.Sound('beep_short.mp3')
        self.snd_fatigue = pygame.mixer.Sound('alarm_loud.mp3')
        
        # Time tracking
        self.eyes_closed_since = None
        self.distracted_since = None
        
        # Thresholds (in seconds)
        self.DROWSY_THRESHOLD = 1.5 
        self.DISTRACT_THRESHOLD = 2.0
        
        # Current State
        self.alert_level = 0 # 0=Normal, 1=Distracted, 2=Fatigued
        
        # UI Overlay
        self.alert_label = tk.Label(self.root, text="SYSTEM ACTIVE", 
                                    font=("Impact", 40), fg="#00FF41", bg="#0B0E14")
        self.alert_label.pack(fill=tk.X, pady=20)
        
    def update_state(self, eyes_state, gaze_direction):
        """
        eyes_state: "OPEN" or "CLOSED"
        gaze_direction: "LEFT", "RIGHT", "CENTER", or " UP"/" DOWN"
        """
        now = time.time()
        
        # --- 1. Process Ewyes (Fatigue overrides everything) ---
        if eyes_state == "CLOSED":
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
        else:
            self.eyes_closed_since = None # Reset on open
            
        # --- 2. Process Gaze (Distraction) ---
        if "CENTER" not in gaze_direction: # If looking Left/Right/Up/Down
            if self.distracted_since is None:
                self.distracted_since = now
        else:
            self.distracted_since = None # Reset on centered
            
        # --- 3. Evaluate Conditions ---
        self._evaluate_alerts(now)

    def _evaluate_alerts(self, now):
        # Check Critical Fatigue First
        if self.eyes_closed_since and (now - self.eyes_closed_since) >= self.DROWSY_THRESHOLD:
            self.trigger_fatigue_alert()
            
        # Check Distraction Second
        elif self.distracted_since and (now - self.distracted_since) >= self.DISTRACT_THRESHOLD:
            self.trigger_distract_alert()
            
        # Reset to Normal
        else:
            self.reset_alerts()

    def trigger_fatigue_alert(self):
        if self.alert_level != 2:
            self.alert_level = 2
            self.alert_label.config(text="!!! WAKE UP !!!", fg="white", bg="#FF0000")
            self.root.config(bg="#FF0000")
            # pygame.mixer.Sound.play(self.snd_fatigue, loops=-1) # Loop alarm
            print("CRITICAL: AUDIO ALARM TRIGGERED")

    def trigger_distract_alert(self):
        if self.alert_level != 1:
            self.alert_level = 1
            self.alert_label.config(text="EYES ON ROAD", fg="black", bg="#FFD700")
            self.root.config(bg="#FFD700")
            # pygame.mixer.Sound.play(self.snd_distract)
            print("WARNING: DISTRACTION BEEP TRIGGERED")

    def reset_alerts(self):
        if self.alert_level != 0:
            self.alert_level = 0
            self.alert_label.config(text="FOCUSED", fg="#00FF41", bg="#0B0E14")
            self.root.config(bg="#0B0E14")
            pygame.mixer.stop() # Stop any playing audio

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root, init_port=None, init_baud=DEFAULT_BAUD):
        self.root    = root
        self.root.title('Camera Stream Viewer')

        self._pending      = None
        self._lock         = threading.Lock()
        self._last_ts      = None
        self._tk_img       = None

        self._wcam_pending = None
        self._wcam_lock    = threading.Lock()
        self._wcam_tk_img  = None

        self._build_ui(init_port, init_baud)

        self.receiver = Receiver(self._on_frame, self._log)
        self.receiver.start()

        self.webcam = WebcamCapture(self._on_webcam_frame, self._log)
        self.webcam.start()

        self.root.after(TICK_MS, self._tick)

        if init_port:
            self._connect(init_port, init_baud)

        self.alert_manager = DriverAlertManager(self.root)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self, init_port, init_baud):
        # ── Top area: controls (left) + canvas (right) ──
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.rowconfigure(0, weight=1)

        self._panel = tk.Frame(top, padx=8, pady=8, width=180)
        self._panel.grid(row=0, column=0, sticky='ns')
        self._panel.grid_propagate(False)
        panel = self._panel

        # Arduino camera
        arduino_frame = tk.Frame(top)
        arduino_frame.grid(row=0, column=1, sticky='nw', padx=4, pady=4)
        tk.Label(arduino_frame, text='Arduino camera', anchor='w').pack(fill=tk.X)
        self._canvas = tk.Canvas(arduino_frame, bg='black', width=1, height=1)
        self._canvas.pack()

        # Laptop webcam
        webcam_frame = tk.Frame(top)
        webcam_frame.grid(row=0, column=2, sticky='nw', padx=4, pady=4)
        tk.Label(webcam_frame, text='Laptop webcam', anchor='w').pack(fill=tk.X)
        self._wcam_canvas = tk.Canvas(webcam_frame, bg='#111', width=320, height=240)
        self._wcam_canvas.pack()

        # ── Log area at the bottom ──
        log_frame = tk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(log_frame, text='Log', anchor='w').pack(side=tk.TOP, fill=tk.X, padx=4)
        self._log_box = scrolledtext.ScrolledText(
            log_frame, height=8, state='disabled',
            font=('Menlo', 10), wrap=tk.WORD)
        self._log_box.pack(fill=tk.X, padx=4, pady=(0, 4))

        # ── Controls ──
        row = 0

        tk.Label(panel, text='Port').grid(row=row, column=0, sticky='w'); row += 1
        ports = [p.device for p in sorted(list_ports.comports())]
        self._port_var = tk.StringVar(value=init_port or (ports[0] if ports else ''))
        ttk.Combobox(panel, textvariable=self._port_var,
                     values=ports, width=20).grid(row=row, column=0, columnspan=2,
                                                   pady=2); row += 1

        tk.Label(panel, text='Baud').grid(row=row, column=0, sticky='w'); row += 1
        self._baud_var = tk.StringVar(value=str(init_baud))
        tk.Entry(panel, textvariable=self._baud_var, width=12).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

        self._btn = tk.Button(panel, text='Connect', command=self._on_connect)
        self._btn.grid(row=row, column=0, columnspan=2, pady=6); row += 1

        self._status_var = tk.StringVar(value='Not connected')
        self._fps_var    = tk.StringVar(value='FPS: —')
        self._res_var    = tk.StringVar(value='Res: —')
        self._gaze_var   = tk.StringVar(value='Gaze: —')
        self._mp_stat_var = tk.StringVar(value='MediaPipe: —')
        for var in [self._status_var, self._fps_var, self._res_var,
                    self._gaze_var, self._mp_stat_var]:
            tk.Label(panel, textvariable=var, anchor='w',
                     wraplength=170).grid(row=row, column=0, columnspan=2,
                                          sticky='w'); row += 1

        ttk.Separator(panel, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=6); row += 1

        tk.Label(panel, text='Resolution').grid(row=row, column=0, sticky='w'); row += 1
        self._res_cmd_var = tk.StringVar(value='QQVGA')
        res_box = ttk.Combobox(panel, textvariable=self._res_cmd_var,
                               values=['QQVGA', 'QVGA', 'CIF'], width=8, state='readonly')
        res_box.grid(row=row, column=0, columnspan=2, sticky='w', pady=2)
        res_box.bind('<<ComboboxSelected>>', self._on_resolution_changed); row += 1

        tk.Label(panel, text='Rotate').grid(row=row, column=0, sticky='w'); row += 1
        self._rot_var = tk.StringVar(value='90')
        ttk.Combobox(panel, textvariable=self._rot_var,
                     values=['0', '90', '180', '270'], width=6,
                     state='readonly').grid(row=row, column=0, columnspan=2,
                                            sticky='w', pady=2); row += 1

        self._mp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text='MediaPipe overlay',
                       variable=self._mp_var).grid(row=row, column=0, columnspan=2,
                                                   sticky='w', pady=(8, 0)); row += 1

        ttk.Separator(panel, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=6); row += 1

        self._wcam_btn = tk.Button(panel, text='Start webcam',
                                   command=self._on_webcam_toggle)
        self._wcam_btn.grid(row=row, column=0, columnspan=2, pady=2); row += 1

    # ── Logging ────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        line = f'[{ts}] {msg}\n'
        # Safe to call from any thread via after()
        self.root.after(0, self._append_log, line)

    def _append_log(self, line: str):
        self._log_box.config(state='normal')
        self._log_box.insert(tk.END, line)
        self._log_box.see(tk.END)
        # Keep log from growing unbounded
        lines = int(self._log_box.index('end-1c').split('.')[0])
        if lines > 500:
            self._log_box.delete('1.0', '100.0')
        self._log_box.config(state='disabled')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _on_connect(self):
        port = self._port_var.get().strip()
        if not port:
            self._status_var.set('Select a port first')
            return
        try:
            baud = int(self._baud_var.get())
        except ValueError:
            self._status_var.set('Invalid baud rate')
            return
        self._connect(port, baud)

    def _connect(self, port, baud):
        try:
            self.receiver.connect(port, baud)
            self._status_var.set(f'Connected: {port}')
            self._btn.config(text='Reconnect')
        except Exception as e:
            self._status_var.set(f'Error: {e}')
            self._log(f'Connection failed: {e}')

    def _on_resolution_changed(self, _event=None):
        cmd = self._res_cmd_var.get()
        self.receiver.send(cmd)

    def _on_webcam_toggle(self):
        if self._wcam_btn.cget('text') == 'Start webcam':
            self.webcam.start_capture()
            self._wcam_btn.config(text='Stop webcam')
        else:
            self.webcam.stop_capture()
            self._wcam_btn.config(text='Start webcam')
            self._wcam_canvas.delete('all')

    def _on_frame(self, img):
        with self._lock:
            self._pending = img

    def _on_webcam_frame(self, img):
        with self._wcam_lock:
            self._wcam_pending = img

    # ── Main loop ──────────────────────────────────────────────────────────

    def _tick(self):
        with self._lock:
            img = self._pending
            self._pending = None

        if img is not None:
            now = time.monotonic()
            if self._last_ts:
                self._fps_var.set(f'FPS: {1.0 / max(now - self._last_ts, 1e-6):.1f}')
            self._last_ts = now
            self._res_var.set(f'Res: {img.width}×{img.height}')
            self._show(img)

        with self._wcam_lock:
            wcam_img = self._wcam_pending
            self._wcam_pending = None

        if wcam_img is not None:
            self._show_webcam(wcam_img)

        self.root.after(TICK_MS, self._tick)

    def _show(self, img):
        # Rotation
        rot = int(self._rot_var.get())
        if rot:
            img = img.rotate(rot, expand=True)

        # MediaPipe
        if self._mp_var.get():
            img, info = annotate(img)
            if info['detected']:
                gaze = info['gaze']
                h, v = info['h_ratio'], info['v_ratio']
                n    = info['num_landmarks']
                self._gaze_var.set(f'Gaze: {gaze}')
                self._mp_stat_var.set(f'MediaPipe: OK ({n} pts)\nh={h:.2f} v={v:.2f}')
                self._log(f'MediaPipe: face detected | {n} landmarks | '
                          f'gaze={gaze} | h={h:.3f} v={v:.3f}')
                self.alert_manager.update_state(info['eyes_state'], info['gaze'])
            else:
                self._gaze_var.set('Gaze: no face')
                self._mp_stat_var.set('MediaPipe: no face detected')
                self._log('MediaPipe: no face detected')
                self.alert_manager.reset_alerts()
        else:
            img = img.convert('RGB')
            self._gaze_var.set('Gaze: (overlay off)')
            self._mp_stat_var.set('MediaPipe: disabled')

        # Each camera gets half the available horizontal space
        avail_w = max(100, (self.root.winfo_width() - self._panel.winfo_width() - 40) // 2)
        avail_h = max(100, self.root.winfo_height() - self._log_box.winfo_height() - 40)
        scale   = min(avail_w / img.width, avail_h / img.height)
        new_w   = max(1, int(img.width  * scale))
        new_h   = max(1, int(img.height * scale))
        img     = img.resize((new_w, new_h), Image.NEAREST)

        self._canvas.config(width=new_w, height=new_h)
        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.delete('all')
        self._canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

    def _show_webcam(self, img):
        if self._mp_var.get():
            img, info = annotate(img)
            if info['detected']:
                self._log(f'Webcam MediaPipe: face detected | gaze={info["gaze"]} | '
                          f'h={info["h_ratio"]:.3f} v={info["v_ratio"]:.3f}')

        avail_w = max(100, (self.root.winfo_width() - self._panel.winfo_width() - 40) // 2)
        avail_h = max(100, self.root.winfo_height() - self._log_box.winfo_height() - 40)
        scale   = min(avail_w / img.width, avail_h / img.height)
        new_w   = max(1, int(img.width  * scale))
        new_h   = max(1, int(img.height * scale))
        img     = img.resize((new_w, new_h), Image.LANCZOS)

        self._wcam_canvas.config(width=new_w, height=new_h)
        self._wcam_tk_img = ImageTk.PhotoImage(img)
        self._wcam_canvas.delete('all')
        self._wcam_canvas.create_image(0, 0, anchor='nw', image=self._wcam_tk_img)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Camera stream viewer with gaze tracking')
    parser.add_argument('--port', default=None)
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    root = tk.Tk()
    App(root, init_port=args.port, init_baud=args.baud)
    root.mainloop()


if __name__ == '__main__':
    main()
