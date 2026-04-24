/*
 * Capture.ino
 *
 * One sketch, two modes:
 *   PREVIEW  -> live QQVGA grayscale stream
 *   FULL     -> QVGA grayscale, one frame per CAPTURE command
 *
 * Binary frame format:
 *   [MAGIC:  0xDE 0xAD 0xBE 0xEF]  4 bytes
 *   [FORMAT: 0x01=GRAYSCALE]       1 byte
 *   [WIDTH:  uint16 little-endian] 2 bytes
 *   [HEIGHT: uint16 little-endian] 2 bytes
 *   [SIZE:   uint32 little-endian] 4 bytes
 *   [PIXELS: SIZE bytes]
 *
 * Serial commands:
 *   MODE PREVIEW
 *   MODE FULL
 *   CAPTURE
 *   STATUS
 */

#include <Arduino_OV767X.h>

#define BAUD_RATE    2000000
#define CAM_TYPE     OV7675
#define PREVIEW_FPS  5
#define FULL_FPS     1
#define PREVIEW_RES  QQVGA
#define FULL_RES     QVGA
#define FULL_FALLBACK_RES QCIF

static const uint8_t MAGIC[] = { 0xDE, 0xAD, 0xBE, 0xEF };

enum CaptureMode {
  MODE_PREVIEW = 0,
  MODE_FULL = 1,
};

static CaptureMode g_mode = MODE_PREVIEW;
static bool g_camera_ready = false;
static bool g_camera_started = false;
static int g_resolution = PREVIEW_RES;

static const char* resolution_label(int resolution) {
  switch (resolution) {
    case VGA: return "VGA";
    case CIF: return "CIF";
    case QVGA: return "QVGA";
    case QCIF: return "QCIF";
    case QQVGA: return "QQVGA";
    default: return "UNKNOWN";
  }
}

static bool init_camera(int resolution, int fps) {
  if (g_camera_started) {
    Camera.end();
    delay(50);
  }

  if (!Camera.begin(resolution, GRAYSCALE, fps, CAM_TYPE)) {
    g_camera_ready = false;
    g_camera_started = false;
    Serial.println("ERR: camera init failed");
    return false;
  }

  g_camera_ready = true;
  g_camera_started = true;
  g_resolution = resolution;
  Serial.print("READY ");
  Serial.print(Camera.width());
  Serial.print("x");
  Serial.print(Camera.height());
  Serial.print(" GRAY ");
  Serial.print(fps);
  Serial.print("fps ");
  Serial.print(resolution_label(resolution));
  Serial.println();
  return true;
}

static void set_mode(CaptureMode mode) {
  if (g_camera_ready && g_mode == mode) {
    return;
  }

  g_mode = mode;

  if (mode == MODE_PREVIEW) {
    if (init_camera(PREVIEW_RES, PREVIEW_FPS)) {
      Serial.println("MODE PREVIEW");
    }
  } else {
    if (init_camera(FULL_RES, FULL_FPS) || init_camera(FULL_FALLBACK_RES, FULL_FPS)) {
      Serial.println("MODE FULL");
    }
  }
}

static void send_status() {
  Serial.print("STATUS ");
  Serial.print(g_mode == MODE_PREVIEW ? "PREVIEW " : "FULL ");
  if (g_camera_ready) {
    Serial.print(Camera.width());
    Serial.print("x");
    Serial.print(Camera.height());
    Serial.print(" ");
    Serial.println(resolution_label(g_resolution));
  } else {
    Serial.println("NOT_READY");
  }
}

static void send_frame() {
  if (!g_camera_ready) {
    Serial.println("ERR: camera not ready");
    return;
  }

  const uint16_t w = (uint16_t)Camera.width();
  const uint16_t h = (uint16_t)Camera.height();
  const uint32_t gray_size = (uint32_t)w * (uint32_t)h;

  // In GRAYSCALE mode the library writes one byte per pixel to the buffer.
  uint8_t* buf = (uint8_t*)malloc(gray_size);
  if (!buf) {
    if (g_mode == MODE_FULL && g_resolution != FULL_FALLBACK_RES) {
      Serial.println("WARN: fallback to QCIF");
      if (init_camera(FULL_FALLBACK_RES, FULL_FPS)) {
        send_frame();
        return;
      }
    }
    Serial.print("ERR: OOM gray=");
    Serial.println(gray_size);
    return;
  }

  Camera.readFrame(buf);

  Serial.write(MAGIC, 4);
  Serial.write((uint8_t)0x01);
  Serial.write((uint8_t)(w & 0xFF));
  Serial.write((uint8_t)((w >> 8) & 0xFF));
  Serial.write((uint8_t)(h & 0xFF));
  Serial.write((uint8_t)((h >> 8) & 0xFF));
  Serial.write((uint8_t)(gray_size & 0xFF));
  Serial.write((uint8_t)((gray_size >> 8) & 0xFF));
  Serial.write((uint8_t)((gray_size >> 16) & 0xFF));
  Serial.write((uint8_t)((gray_size >> 24) & 0xFF));
  Serial.write(buf, gray_size);
  Serial.flush();

  free(buf);
}

static void handle_command(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.length() == 0) {
    return;
  }

  if (cmd == "MODE PREVIEW" || cmd == "PREVIEW") {
    set_mode(MODE_PREVIEW);
  } else if (cmd == "MODE FULL" || cmd == "FULL") {
    set_mode(MODE_FULL);
  } else if (cmd == "CAPTURE") {
    if (g_mode != MODE_FULL) {
      set_mode(MODE_FULL);
    }
    send_frame();
  } else if (cmd == "STATUS") {
    send_status();
  } else {
    Serial.print("ERR: unknown cmd ");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(50);
  while (!Serial) { }
  delay(500);
  set_mode(MODE_PREVIEW);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handle_command(cmd);
  }

  if (g_mode == MODE_PREVIEW && g_camera_ready) {
    send_frame();
  }
}
