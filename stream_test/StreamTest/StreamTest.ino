#include <Arduino_OV767X.h>
#include "base64.h"

// --- Configuration ---
#define BAUD_RATE  2000000
#define CAM_TYPE   OV7675
#define CAM_FORMAT RGB565
#define CAM_FPS    5

// EIML framing
static const int EIML_HEADER_SIZE = 12;
static const int EIML_SOF_SIZE    = 3;
static const uint8_t EIML_SOF[]   = { 0xFF, 0xA0, 0xFF };

typedef enum { EIML_GRAYSCALE = 1, EIML_RGB888 = 2 } EimlFormat;
typedef struct { uint8_t format; uint32_t width; uint32_t height; } EimlHeader;

// Current resolution (can be changed at runtime via serial command)
static int g_resolution = QQVGA;

static void build_header(EimlHeader h, uint8_t *out) {
  out[0] = EIML_SOF[0]; out[1] = EIML_SOF[1]; out[2] = EIML_SOF[2];
  out[3] = h.format;
  for (int i = 0; i < 4; i++) out[4 + i] = (h.width  >> (i * 8)) & 0xFF;
  for (int i = 0; i < 4; i++) out[8 + i] = (h.height >> (i * 8)) & 0xFF;
}

static void init_camera(int resolution) {
  g_resolution = resolution;
  if (!Camera.begin(g_resolution, CAM_FORMAT, CAM_FPS, CAM_TYPE)) {
    Serial.println("ERR: camera init failed");
    return;
  }
  Serial.print("READY ");
  Serial.print(Camera.width());
  Serial.print("x");
  Serial.println(Camera.height());
}

// Check for incoming resolution commands: "QQVGA", "QVGA", "CIF"
static void check_commands() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if      (cmd == "QQVGA") init_camera(QQVGA);
  else if (cmd == "QVGA")  init_camera(QVGA);
  else if (cmd == "CIF")   init_camera(CIF);
}

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(50);
  while (!Serial);
  delay(500);
  init_camera(QQVGA);
}

void loop() {
  check_commands();

  int w   = Camera.width();
  int h   = Camera.height();
  int bpp = Camera.bytesPerPixel();

  uint8_t *raw = (uint8_t *)malloc(w * h * bpp);
  if (!raw) { Serial.println("ERR: OOM raw"); return; }
  Camera.readFrame(raw);

  // Convert RGB565 → RGB888
  int px_count = w * h;
  uint8_t *rgb = (uint8_t *)malloc(px_count * 3);
  if (!rgb) { free(raw); Serial.println("ERR: OOM rgb"); return; }
  // OV767X sends RGB565 little-endian (low byte first, high byte second)
  for (int i = 0; i < px_count; i++) {
    uint8_t lo = raw[2 * i];
    uint8_t hi = raw[2 * i + 1];
    rgb[3 * i]     = hi & 0xF8;                               // R
    rgb[3 * i + 1] = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3); // G
    rgb[3 * i + 2] = (lo & 0x1F) << 3;                        // B
  }
  free(raw);

  uint8_t hdr_buf[EIML_HEADER_SIZE];
  EimlHeader hdr = { EIML_RGB888, (uint32_t)w, (uint32_t)h };
  build_header(hdr, hdr_buf);

  uint8_t enc_hdr[(EIML_HEADER_SIZE + 2) / 3 * 4 + 1];
  encode_base64(hdr_buf, EIML_HEADER_SIZE, enc_hdr);

  uint32_t rgb_bytes   = px_count * 3;
  uint32_t enc_px_len  = (rgb_bytes + 2) / 3 * 4;
  uint8_t *enc_px = (uint8_t *)malloc(enc_px_len + 1);
  if (!enc_px) { free(rgb); Serial.println("ERR: OOM enc"); return; }
  encode_base64(rgb, rgb_bytes, enc_px);
  free(rgb);

  Serial.print((char *)enc_hdr);
  Serial.println((char *)enc_px);
  free(enc_px);
}
