#define POT_PIN A0       // Potentiometer center pin
#define PRESSURE_PIN A1  // Pressure pin

double pos_analog = 0;
double pos_degree = 0;
double old_pos = 0;


void setup() {
  pinMode(POT_PIN, INPUT); // Set potentiometer to input
  pinMode(PRESSURE_PIN, INPUT); // Set potentiometer to input

  Serial.begin(9600); // open the serial port at 9600 bps:
}

void loop() {
  pos_analog = analogRead(POT_PIN);         // Range: 0–1023
  pressure_analog = analogRead(PRESSURE_PIN)

  Serial.println(pos_analog);
  Serial.println(pressure_analog);

  
}
