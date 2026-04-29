#include <Arduino_LPS22HB.h>
#include <Arduino_APDS9960.h>
#include <Wire.h>
#include "MAX86916_eda.h"

// Initialize the oximeter object
MAX86916_eda ppg;

void setup() {
  Serial.begin(115200);
  Wire.begin(); // Required for the oximeter's I2C communication

  // ---------------------------------------------------------
  // 1. Initialize LPS22HB pressure sensor
  // ---------------------------------------------------------
  if (!BARO.begin()) {
    Serial.println("Failed to initialize pressure sensor!");
    while (1);
  }
  Serial.println("Pressure sensor initialized successfully!");

  // ---------------------------------------------------------
  // 2. Initialize APDS9960 proximity sensor
  // ---------------------------------------------------------
  if (!APDS.begin()) {
    Serial.println("Failed to initialize proximity sensor!");
    while (1);
  }
  Serial.println("Proximity sensor initialized successfully!");

  // ---------------------------------------------------------
  // 3. Initialize MAX86916 Oximeter
  // ---------------------------------------------------------
  ppg.begin();
  ppg.setup();
  ppg.setPulseAmplitudeRed(0x7F);
  ppg.setPulseAmplitudeIR(0x7F);
  ppg.setPulseAmplitudeGreen(0x7F);
  ppg.setPulseAmplitudeBlue(0x7F);
  Serial.println("Oximeter initialized successfully!");
  
  // Add a blank line before the main output starts
  Serial.println(); 
}

void loop() {
  // 1. Read Pressure
  float pressure = BARO.readPressure();

  // 2. Read Proximity (Default to -1 if not ready)
  int proximity = -1; 
  if (APDS.proximityAvailable()) {
    proximity = APDS.readProximity();
  }

  // 3. Read Oximeter values (non-blocking FIFO mode)
  //    Instead of calling blocking getIR()/getRed(), we use check()
  //    which pulls all available FIFO data into local buffers at once.
  ppg.check(); // Non-blocking: reads all available samples from sensor FIFO

  // 4. Output all buffered samples
  //    This ensures no sample is discarded — every PPG reading is sent to the PC.
  while (ppg.available()) {
    uint32_t red_val = ppg.getFIFORed();
    uint32_t ir_val  = ppg.getFIFOIR();

    // Print all values in a single comma-separated line for Python
    // Format: Pressure, Proximity, Red, IR
    Serial.print(pressure);
    Serial.print(",");
    Serial.print(proximity);
    Serial.print(",");
    Serial.print(red_val);
    Serial.print(",");
    Serial.println(ir_val);

    ppg.nextSample(); // Advance FIFO read pointer
  }

  // No delay needed — the PPG sensor's internal sampling rate dictates timing.
  // The loop runs as fast as possible, and the Python side receives every sample.
}
