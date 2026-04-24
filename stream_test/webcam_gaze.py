"""
webcam_gaze.py — MediaPipe iris/gaze test using the laptop webcam.

Requirements:
    pip install opencv-python mediapipe numpy

Usage:
    python webcam_gaze.py
    Press Q to quit.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'face_landmarker/face_landmarker/float16/1/face_landmarker.task')

if not MODEL_PATH.exists():
    print(f'Downloading face landmarker model → {MODEL_PATH} ...')
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print('Done.')

opts = _mp_vision.FaceLandmarkerOptions(
    base_options=_mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = _mp_vision.FaceLandmarker.create_from_options(opts)

# ── Landmark indices ────────────────────────────────────────────────────────
LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_IRIS  = [468,469,470,471,472]
RIGHT_IRIS = [473,474,475,476,477]

L_OUTER, L_INNER, L_TOP, L_BOT = 33,  133, 159, 145
R_OUTER, R_INNER, R_TOP, R_BOT = 263, 362, 386, 374
L_CTR, R_CTR                    = 468, 473

# ── Colours (BGR for OpenCV) ────────────────────────────────────────────────
EYE_CLR  = (255, 220,   0)   # cyan
IRIS_CLR = (  0, 200, 255)   # amber
TEXT_CLR = (  0, 255,   0)   # green


def gaze_direction(lm):
    def x(i): return lm[i].x
    def y(i): return lm[i].y

    h_l = (x(L_CTR) - x(L_OUTER)) / max(x(L_INNER) - x(L_OUTER), 1e-6)
    h_r = (x(R_CTR) - x(R_INNER)) / max(x(R_OUTER) - x(R_INNER), 1e-6)
    h   = (h_l + h_r) / 2.0

    v_l = (y(L_CTR) - y(L_TOP)) / max(y(L_BOT) - y(L_TOP), 1e-6)
    v_r = (y(R_CTR) - y(R_TOP)) / max(y(R_BOT) - y(R_TOP), 1e-6)
    v   = (v_l + v_r) / 2.0

    h_label = 'LEFT'   if h < 0.38 else 'RIGHT' if h > 0.62 else 'CENTER'
    v_label = ' UP'    if v < 0.35 else ' DOWN'  if v > 0.65 else ''
    return h_label + v_label, round(h, 3), round(v, 3)


def px(lm_pt, w, h):
    return (int(lm_pt.x * w), int(lm_pt.y * h))


def draw_overlay(frame, lm):
    h, w = frame.shape[:2]

    # Eye contour dots
    for idx in LEFT_EYE + RIGHT_EYE:
        cv2.circle(frame, px(lm[idx], w, h), 2, EYE_CLR, -1)

    # Iris rings + centre dot
    for iris_ids, ctr_id in ((LEFT_IRIS, L_CTR), (RIGHT_IRIS, R_CTR)):
        pts = [px(lm[i], w, h) for i in iris_ids]
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        cx, cy = px(lm[ctr_id], w, h)
        rx = max(1, (max(xs) - min(xs)) // 2)
        ry = max(1, (max(ys) - min(ys)) // 2)
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, IRIS_CLR, 2)
        cv2.circle(frame, (cx, cy), 2, IRIS_CLR, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: could not open webcam')
        return

    print('Webcam open. Press Q to quit.')
    fps_time = time.monotonic()
    fps      = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.face_landmarks:
            lm   = result.face_landmarks[0]
            gaze, h_ratio, v_ratio = gaze_direction(lm)
            draw_overlay(frame, lm)

            cv2.putText(frame, f'Gaze: {gaze}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_CLR, 2)
            cv2.putText(frame, f'h={h_ratio:.2f}  v={v_ratio:.2f}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_CLR, 1)
            print(f'\rGaze: {gaze:<16} h={h_ratio:.3f}  v={v_ratio:.3f}   ', end='')
        else:
            cv2.putText(frame, 'No face detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print('\rNo face detected                           ', end='')

        # FPS counter
        now   = time.monotonic()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-6))
        fps_time = now
        cv2.putText(frame, f'FPS: {fps:.1f}',
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('Webcam Gaze Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == '__main__':
    main()
