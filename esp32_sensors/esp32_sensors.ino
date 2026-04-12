/*
 * SCENTATION — SENSOR NODE (ESP32 DevKit V1)
 * GSR: GPIO34 (ADC)  |  MAX30102: SDA 21, SCL 22  |  LED: GPIO2
 * Serial 115200: GSR:x,HR:y,HRV:z
 *
 * Install: SparkFun MAX3010x Pulse and Proximity Sensor Library
 */

// #define MOCK_MODE

#ifndef MOCK_MODE
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#endif

#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

const int PIN_GSR = 34;
const int PIN_LED = 2;
const int PIN_SDA = 21;
const int PIN_SCL = 22;

const unsigned long SERIAL_BAUD = 115200;
const unsigned long SAMPLE_INTERVAL_MS = 250;
const unsigned long LED_BLINK_MS = 500;

// 12-bit ADC after analogReadResolution(12) — matches ESP32 Arduino core default range 0..4095.
const float GSR_ADC_MAX = (float)((1 << 12) - 1);
const float GSR_R_REF = 10000.0f;
const float GSR_EMA_ALPHA = 0.3f;
const int GSR_PEAK_BUF = 20;

const int PPG_BEAT_BUF = 20;
const int PPG_MIN_BEATS = 3;
const float PPG_STALE_SEC = 3.0f;

float gsrFiltered = 0.0f;
bool gsrFirst = true;
float gsrPeakBuf[GSR_PEAK_BUF];
int gsrPeakIdx = 0;

float currentHR = -1.0f;
float currentHRV = -1.0f;
bool ppgOnline = false;
unsigned long lastBeatMs = 0;
unsigned long beatTs[PPG_BEAT_BUF];
int beatWrite = 0;
int beatCount = 0;

#ifndef MOCK_MODE
MAX30105 particleSensor;
#endif

unsigned long lastSample = 0;
unsigned long lastBlink = 0;
bool ledOn = false;

#ifdef MOCK_MODE
unsigned long mockT0 = 0;
float mockHR = 72.0f;
float mockHRV = 42.0f;
#endif

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {}
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_GSR, INPUT);
  // Match GSR_ADC_MAX to Seeed Grove GSR output on SIG (~0–3.3 V on GPIO34). Adjust if wiring differs.
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  for (int i = 0; i < GSR_PEAK_BUF; i++) gsrPeakBuf[i] = 3.0f;
  for (int i = 0; i < PPG_BEAT_BUF; i++) beatTs[i] = 0;

#ifdef MOCK_MODE
  mockT0 = millis();
  ppgOnline = true;
  Serial.println("SCENTATION_SENSOR_NODE_v1.0|GSR:MOCK|PPG:MOCK|READY");
#else
  Wire.begin(PIN_SDA, PIN_SCL);
  // Limit blocking on clock stretch / stuck slave (ESP32 Arduino Wire; ms).
  Wire.setTimeOut(50);
  ppgOnline = particleSensor.begin(Wire, I2C_SPEED_FAST);
  if (ppgOnline) {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    particleSensor.setPulseAmplitudeIR(0x1F);
    Serial.println("SCENTATION_SENSOR_NODE_v1.0|GSR:OK|PPG:OK|READY");
  } else {
    Serial.println("SCENTATION_SENSOR_NODE_v1.0|GSR:OK|PPG:FAIL|DEGRADED");
  }
#endif
  lastSample = millis();
  lastBlink = millis();
}

#ifndef MOCK_MODE
// Rate-limited re-init after ppgOnline drops (e.g. FIFO budget fail-safe).
const unsigned long PPG_REINIT_INTERVAL_MS = 30000UL;
unsigned long lastPpgReinitAttemptMs = 0;

void tryReinitPPG() {
  unsigned long now = millis();
  if (lastPpgReinitAttemptMs != 0 && (now - lastPpgReinitAttemptMs) < PPG_REINIT_INTERVAL_MS) {
    return;
  }
  lastPpgReinitAttemptMs = now;
  Wire.setTimeOut(50);
  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    particleSensor.setPulseAmplitudeIR(0x1F);
    ppgOnline = true;
    lastBeatMs = 0;
    beatWrite = 0;
    beatCount = 0;
    for (int i = 0; i < PPG_BEAT_BUF; i++) beatTs[i] = 0;
    currentHR = currentHRV = -1.0f;
  }
}
#endif

void loop() {
  unsigned long now = millis();
#ifndef MOCK_MODE
  if (!ppgOnline) {
    tryReinitPPG();
  }
#endif
  if (now - lastBlink >= LED_BLINK_MS) {
    ledOn = !ledOn;
    digitalWrite(PIN_LED, ledOn);
    lastBlink = now;
  }

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
#ifdef MOCK_MODE
    mockGSR(now);
    updateMockPPG(now);
#else
    updateGSR();
    if (ppgOnline) updatePPG();
#endif
    sendLine();
  }
}

void updateGSR() {
  int raw = analogRead(PIN_GSR);
  float cond;
  // Conductance (µS) from divider model with R_ref — calibrate against known R if needed.
  if (raw <= 0) cond = 0.0f;
  else if (raw >= (int)GSR_ADC_MAX) cond = 50.0f;
  else cond = (GSR_ADC_MAX - (float)raw) / (GSR_R_REF * (float)raw) * 1000000.0f;
  cond = constrain(cond, 0.0f, 30.0f);
  if (gsrFirst) {
    gsrFiltered = cond;
    gsrFirst = false;
  } else {
    gsrFiltered = GSR_EMA_ALPHA * cond + (1.0f - GSR_EMA_ALPHA) * gsrFiltered;
  }
  gsrPeakBuf[gsrPeakIdx] = gsrFiltered;
  gsrPeakIdx = (gsrPeakIdx + 1) % GSR_PEAK_BUF;
}

void updatePPG() {
  int n = 0;
  unsigned long t0 = micros();
  // Slightly generous budget reduces false "PPG offline" when the FIFO is deep (~4 Hz line rate).
  const unsigned long kPpgBudgetUs = 12000;
  bool hitBudget = false;
  // Cap FIFO drain per loop tick so ~4 Hz GSR/line cadence is not starved.
  while (particleSensor.available() && n < 48) {
    if (micros() - t0 > kPpgBudgetUs) {
      hitBudget = true;
      break;
    }
    long ir = particleSensor.getIR();
    if (checkForBeat(ir)) {
      unsigned long t = millis();
      if (lastBeatMs > 0) {
        unsigned long rr = t - lastBeatMs;
        if (rr >= 300 && rr <= 1500) {
          beatTs[beatWrite] = t;
          beatWrite = (beatWrite + 1) % PPG_BEAT_BUF;
          if (beatCount < PPG_BEAT_BUF) beatCount++;
        }
      }
      lastBeatMs = t;
    }
    particleSensor.nextSample();
    n++;
  }
  static uint8_t ppgBudgetStreak = 0;
  if (hitBudget) {
    if (++ppgBudgetStreak >= 50) {
      ppgOnline = false;
      currentHR = currentHRV = -1.0f;
      ppgBudgetStreak = 0;
    }
  } else {
    ppgBudgetStreak = 0;
  }
  computeHRHRV();
}

/*
 * QA / device validation (post-demo): compare HR/HRV vs reference belt or known PPG trace;
 * verify beatTs ring indexing under arrhythmia and boundary wrap. See project review checklist.
 */
void computeHRHRV() {
  unsigned long now = millis();
  if (beatCount < PPG_MIN_BEATS) {
    currentHR = currentHRV = -1.0f;
    return;
  }
  unsigned long last = beatTs[(beatWrite + PPG_BEAT_BUF - 1) % PPG_BEAT_BUF];
  if (now - last > (unsigned long)(PPG_STALE_SEC * 1000.0f)) {
    currentHR = currentHRV = -1.0f;
    return;
  }
  int n = min(beatCount, PPG_BEAT_BUF);
  float rrMs[PPG_BEAT_BUF];
  int rrN = 0;
  int start = (beatWrite - n + PPG_BEAT_BUF) % PPG_BEAT_BUF;
  for (int i = 0; i < n - 1; i++) {
    int i1 = (start + i) % PPG_BEAT_BUF;
    int i2 = (start + i + 1) % PPG_BEAT_BUF;
    long d = (long)(beatTs[i2] - beatTs[i1]);
    if (d >= 300 && d <= 1500) rrMs[rrN++] = (float)d;
  }
  if (rrN < 1) {
    currentHR = currentHRV = -1.0f;
    return;
  }
  int hw = min(rrN, 5);
  float sum = 0;
  for (int i = rrN - hw; i < rrN; i++) sum += rrMs[i];
  float avgRR = sum / (float)hw;
  currentHR = 60000.0f / avgRR;

  int hw2 = min(rrN, 10);
  if (hw2 < 3) {
    currentHRV = -1.0f;
    return;
  }
  float ssd = 0;
  int dc = 0;
  for (int i = rrN - hw2; i < rrN - 1; i++) {
    float df = rrMs[i + 1] - rrMs[i];
    ssd += df * df;
    dc++;
  }
  currentHRV = sqrtf(ssd / (float)dc);
}

void sendLine() {
  Serial.print("GSR:");
  Serial.print(gsrFiltered, 2);
  Serial.print(",HR:");
  if (ppgOnline) Serial.print(currentHR, 1);
  else Serial.print("-1.0");
  Serial.print(",HRV:");
  if (ppgOnline) Serial.print(currentHRV, 1);
  else Serial.print("-1.0");
  Serial.println();
}

#ifdef MOCK_MODE
void mockGSR(unsigned long now) {
  float t = (now - mockT0) / 1000.0f;
  float base = 3.0f + 0.5f * sinf(t * 0.15f);
  float sp = 0.0f;
  float ph = fmod(t, 30.0f);
  if (ph < 2.0f) sp = 3.0f * expf(-ph * 1.5f) * (1.0f - expf(-ph * 8.0f));
  gsrFiltered = base + sp + random(-15, 16) / 200.0f;
  gsrFiltered = constrain(gsrFiltered, 0.5f, 15.0f);
}

void updateMockPPG(unsigned long now) {
  (void)now;
  mockHR += random(-100, 101) / 100.0f;
  mockHR = constrain(mockHR, 55.0f, 100.0f);
  mockHR += (72.0f - mockHR) * 0.02f;
  mockHRV = 40.0f + random(-30, 31) / 10.0f;
  mockHRV = constrain(mockHRV, 15.0f, 65.0f);
  currentHR = mockHR;
  currentHRV = mockHRV;
}
#endif
