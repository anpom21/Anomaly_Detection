#define POT_PIN A0       // 电位器
#define PRESSURE_PIN A1  // 压力传感器

void setup() {
  pinMode(POT_PIN, INPUT);
  pinMode(PRESSURE_PIN, INPUT);
  Serial.begin(9600);
}

void loop() {
  int pot_raw = analogRead(POT_PIN);
  int pressure_raw = analogRead(PRESSURE_PIN);

  double position = map(pot_raw, 0, 1023, 0, 300);        // 角度 (0–300 度)
  double pressure = (pressure_raw / 1023.0) * 5.0;        // 压力 (0–5 bar)

  Serial.print("pressure:");
  Serial.print(pressure, 2);
  Serial.print(",position:");
  Serial.println(position, 2);

  delay(100);
}
