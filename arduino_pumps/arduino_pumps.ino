/*
 * SCENTATION — PUMP OUTPUT NODE (Arduino Uno R3)
 *
 * Serial 115200:
 *   PUMP:X,Y,Z  PUMP:ALL  PUMP:OFF  PUMP:TEST  STATUS  PING
 *
 * Relays ACTIVE LOW: LOW = pump ON, HIGH = OFF.
 *
 *        Arduino Uno                    Relay module
 *        5V  --------------------------- VCC
 *        GND --------------------------- GND
 *        5,6,7,8 ----------------------- IN1..IN4
 *        USB --------------------------- Laptop
 *
 * Pumps use EXTERNAL 5V supply — not Arduino 5V rail.
 */

// #define MOCK_MODE
// Bench only: define to enable PUMP:TEST (uses multi-second delay() — stalls serial and auto-shutoff).
// #define ALLOW_PUMP_TEST

#include <string.h>

const int NUM_PUMPS = 4;
const int PUMP_PINS[NUM_PUMPS] = {5, 6, 7, 8};
const int PIN_LED = 13;

const unsigned long SERIAL_BAUD = 115200;
const int CMD_BUFFER_SIZE = 64;
const unsigned long AUTO_SHUTOFF_MS = 30000;
const unsigned long DEBOUNCE_MS = 100;
const unsigned long LED_IDLE_MS = 1000;
const unsigned long LED_ACTIVE_MS = 200;
const unsigned long TEST_PUMP_DURATION_MS = 1000;

bool pumpState[NUM_PUMPS] = {false, false, false, false};
unsigned long lastCommandTime = 0;
unsigned long lastPumpOnTime = 0;
unsigned long lastBlinkTime = 0;
bool ledState = false;
bool testModeRunning = false;
char cmdBuffer[CMD_BUFFER_SIZE];
int cmdIndex = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {}
  for (int i = 0; i < NUM_PUMPS; i++) {
    pinMode(PUMP_PINS[i], OUTPUT);
    digitalWrite(PUMP_PINS[i], HIGH);
  }
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  lastCommandTime = millis();
  lastBlinkTime = millis();
  Serial.println("SCENTATION_PUMP_NODE_v1.0|PUMPS:4|READY");
#ifdef MOCK_MODE
  Serial.println("[MOCK] No relay GPIO toggling");
#endif
}

void loop() {
  unsigned long now = millis();
  readSerialCommand();
  checkAutoShutoff(now);
  updateLED(now);
}

void readSerialCommand() {
  static bool discardUntilNl = false;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      discardUntilNl = false;
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        processCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (c == '\r') {
    } else if (discardUntilNl) {
      // Drop remainder of over-long line (already emitted ERROR:LINE_TOO_LONG).
    } else if (c == 0x7F || c == 0x08) {
      if (cmdIndex > 0) cmdIndex--;
    } else if (cmdIndex < CMD_BUFFER_SIZE - 1) {
      cmdBuffer[cmdIndex++] = c;
    } else {
      discardUntilNl = true;
      cmdIndex = 0;
      Serial.println("ERROR:LINE_TOO_LONG");
    }
  }
}

void processCommand(const char* cmd) {
  const char* p = cmd;
  while (*p == ' ' || *p == '\t') p++;

  // Never debounce pump or safety/diagnostic commands — host may send PUMP lines in quick succession.
  bool noDebounce =
      (strcmp(p, "PING") == 0) || (strcmp(p, "STATUS") == 0) || (strncmp(p, "PUMP:", 5) == 0);

  unsigned long now = millis();
  if (!noDebounce && (now - lastCommandTime < DEBOUNCE_MS)) return;
  lastCommandTime = now;

  if (strcmp(p, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(p, "STATUS") == 0) {
    Serial.print("STATE:");
    for (int i = 0; i < NUM_PUMPS; i++) {
      Serial.print(pumpState[i] ? 1 : 0);
      if (i < NUM_PUMPS - 1) Serial.print(",");
    }
    Serial.println();
  } else if (strncmp(p, "PUMP:", 5) == 0) {
    const char* args = p + 5;
    if (strcmp(args, "OFF") == 0) {
      allPumpsOff();
      Serial.println("ACK:OFF");
    } else if (strcmp(args, "ALL") == 0) {
      for (int i = 0; i < NUM_PUMPS; i++) setPump(i, true);
      lastPumpOnTime = millis();
      Serial.println("ACK:PUMP:1,2,3,4");
    } else if (strcmp(args, "TEST") == 0) {
#ifdef ALLOW_PUMP_TEST
      if (!testModeRunning) runPumpTest();
      else Serial.println("ERROR:TEST_RUNNING");
#else
      Serial.println("ERROR:PUMP_TEST_DISABLED");
#endif
    } else {
      int requested[NUM_PUMPS];
      int n = parsePumpNumbers(args, requested, NUM_PUMPS);
      if (n == 0) {
        Serial.println("ERROR:INVALID_PUMP");
      } else {
        bool ok = true;
        for (int i = 0; i < n; i++) {
          if (requested[i] < 1 || requested[i] > NUM_PUMPS) ok = false;
        }
        if (!ok) Serial.println("ERROR:INVALID_PUMP");
        else {
          allPumpsOff();
          for (int i = 0; i < n; i++) setPump(requested[i] - 1, true);
          lastPumpOnTime = millis();
          Serial.print("ACK:PUMP:");
          for (int i = 0; i < n; i++) {
            Serial.print(requested[i]);
            if (i < n - 1) Serial.print(",");
          }
          Serial.println();
        }
      }
    }
  } else {
    Serial.print("ERROR:UNKNOWN_CMD:");
    Serial.println(p);
  }
}

int parsePumpNumbers(const char* str, int* result, int maxNum) {
  int count = 0;
  const char* ptr = str;
  while (*ptr && count < maxNum) {
    while (*ptr == ' ' || *ptr == ',' || *ptr == '\t') ptr++;
    if (*ptr == '\0') break;
    if (*ptr >= '0' && *ptr <= '9') {
      result[count] = 0;
      while (*ptr >= '0' && *ptr <= '9') {
        result[count] = result[count] * 10 + (*ptr - '0');
        ptr++;
      }
      count++;
    } else {
      while (*ptr && *ptr != ',' && *ptr != ' ') ptr++;
    }
  }
  return count;
}

void setPump(int idx, bool on) {
  if (idx < 0 || idx >= NUM_PUMPS) return;
  bool prev = pumpState[idx];
  pumpState[idx] = on;
#ifndef MOCK_MODE
  digitalWrite(PUMP_PINS[idx], on ? LOW : HIGH);
#endif
  if (prev != on) {
    Serial.print("LOG:PUMP_");
    Serial.print(idx + 1);
    Serial.print(on ? ":ON|T:" : ":OFF|T:");
    Serial.println(millis());
  }
}

void allPumpsOff() {
  for (int i = 0; i < NUM_PUMPS; i++) setPump(i, false);
}

// WARNING: long delay() — do not invoke PUMP:TEST during a live demo (see tools/README.md).
void runPumpTest() {
  testModeRunning = true;
  Serial.println("LOG:TEST_START");
  for (int i = 0; i < NUM_PUMPS; i++) {
    Serial.print("LOG:TEST_PUMP_");
    Serial.println(i + 1);
    setPump(i, true);
    delay(TEST_PUMP_DURATION_MS);
    setPump(i, false);
    delay(200);
  }
  Serial.println("LOG:TEST_COMPLETE");
  testModeRunning = false;
}

void checkAutoShutoff(unsigned long now) {
  if (lastPumpOnTime == 0) return;
  bool any = false;
  for (int i = 0; i < NUM_PUMPS; i++) {
    if (pumpState[i]) any = true;
  }
  if (any && (now - lastPumpOnTime > AUTO_SHUTOFF_MS)) {
    allPumpsOff();
    lastPumpOnTime = 0;
    Serial.println("AUTO_SHUTOFF");
    Serial.println("LOG:SAFETY:PUMPS_FORCE_OFF");
  }
}

void updateLED(unsigned long now) {
  bool any = false;
  for (int i = 0; i < NUM_PUMPS; i++) {
    if (pumpState[i]) any = true;
  }
  unsigned long period = any ? LED_ACTIVE_MS : LED_IDLE_MS;
  if (now - lastBlinkTime >= period) {
    ledState = !ledState;
    digitalWrite(PIN_LED, ledState);
    lastBlinkTime = now;
  }
}
