#define trigPin 4
#define echoPin 7
// #include <time.h>
long duration;
int distance;
long s, e, s1, e1,s2,e2;
String pack;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  // s = millis();
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // s1 = millis();
  duration = pulseIn(echoPin, HIGH);
  // e1 = millis();
  distance = duration * 0.034 / 2;
  // distance = 25;
  // pack = "";
  // for (int i = 0; i < 25; ++i) {
  //   pack += String(distance) + " ";
  //   // Serial.print(distance);
  //   // Serial.print(" ");
  // }
  
  // e = millis();
  Serial.println(distance);
  // s2 = millis();
  // Serial.print(" ");
  // Serial.print(e-s);
  // Serial.print(" ");
  // Serial.print(e1-s1);
  // Serial.print(" ");
  // Serial.print(s1-s);
  // Serial.print(" ");
  // Serial.print(e-e1);
  // Serial.print(" ");
  // Serial.println(pack);
  // e2 = millis();
  // Serial.print(" ");
  // Serial.println(e2-s2);
  

  // delay(50);
}
