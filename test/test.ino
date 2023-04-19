#define trigPin 4
#define echoPin 5

void setup()
{
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);
    Serial.begin(9600);
}

void loop()
{
    Serial.println("haha");
    delay(1000);
}
