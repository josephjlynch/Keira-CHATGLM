/*
 * Uno + CheezPPG + GSR → hub-compatible serial (115200)
 *
 * Prints one line every HUB_LINE_MS to match ~4 Hz expected by
 * scentsation_hub / collect_labeled_data / build_custom_6d:
 *   GSR:<float>,HR:<float>,HRV:<float>
 *
 * Install the CheezPPG library in Arduino IDE (Sketch → Include Library →
 * Add .ZIP Library, or your vendor instructions) so #include "CheezPPG.h" works.
 *
 * Pins (change if your wiring differs):
 *   PPG / pulse  → A0
 *   GSR          → A1
 *
 * Retrain ML if raw GSR scale differs from older datasets.
 */

#include "CheezPPG.h"

#define PPG_INPUT_PIN A0
#define GSR_INPUT_PIN A1
#define LED_PIN 13
#define SAMPLE_RATE 125
#define HUB_LINE_MS 250

CheezPPG ppg(PPG_INPUT_PIN, SAMPLE_RATE);

int gsrThreshold = 0;
unsigned long ledTurnOffTime = 0;
unsigned long lastHubMs = 0;

float lastHr = -1.0f;
float lastHrv = -1.0f;
int lastGsr = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  pinMode(LED_PIN, OUTPUT);

  ppg.setWearThreshold(8);

  Serial.println(F("Calibrating GSR... keep still."));
  long sum = 0;
  for (int i = 0; i < 500; i++) {
    sum += analogRead(GSR_INPUT_PIN);
    delay(5);
  }
  gsrThreshold = sum / 500;
  Serial.println(F("Calibration done. Streaming GSR:,HR:,HRV: ~4Hz."));
  lastHubMs = millis();
}

void loop() {
  unsigned long now = millis();

  if (ppg.checkSampleInterval()) {
    ppg.ppgProcess();
    float hr = ppg.getPpgHr();
    float hrv = ppg.getPpgHrv();
    int gsrValue = analogRead(GSR_INPUT_PIN);

    if (hr != hr || hr < 0.0f) {
      hr = -1.0f;
    }
    if (hrv != hrv || hrv < 0.0f) {
      hrv = -1.0f;
    }

    lastHr = hr;
    lastHrv = hrv;
    lastGsr = gsrValue;

    int diff = abs(gsrThreshold - gsrValue);
    if (diff > 60) {
      digitalWrite(LED_PIN, HIGH);
      ledTurnOffTime = now + 3000UL;
    }
  }

  if (now > ledTurnOffTime && ledTurnOffTime != 0) {
    digitalWrite(LED_PIN, LOW);
    ledTurnOffTime = 0;
  }

  if (now - lastHubMs >= HUB_LINE_MS) {
    lastHubMs = now;
    Serial.print(F("GSR:"));
    Serial.print((float)lastGsr, 2);
    Serial.print(F(",HR:"));
    Serial.print(lastHr, 2);
    Serial.print(F(",HRV:"));
    Serial.println(lastHrv, 2);
  }
}
