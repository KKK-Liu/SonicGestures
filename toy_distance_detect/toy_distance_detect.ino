#define trigPin 4
#define echoPin 7
long duration;
int distance;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;
  for (int i = 0; i < 24; ++i) {
    Serial.print(distance);
    Serial.print(" ");
  }
  Serial.println(distance);

  // delay(50);
}
