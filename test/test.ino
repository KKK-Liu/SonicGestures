#define trigPin 4
#define echoPin 5

void setup()
{
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
    Serial.begin(9600);
    Serial.println(">>>>Start>>>>");
    delay(1000);
}

void loop()
{
    Serial.println("action: empty");
    delay(100);
}
