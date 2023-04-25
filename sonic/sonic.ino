#define selPin0 2
#define selPin1 3
#define selPin2 4
#define selPin3 5
#define selPin4 6

#define enable 7
#define echoPin 8
#define trigPin 9

#define enableTrue false
#define enableFalse true

int sonic_pos;
long duration;
int distance;

int sels[5] = {
    selPin0,
    selPin1,
    selPin2,
    selPin3,
    selPin4,
};
String pack;

void setup()
{
    pinMode(selPin0, OUTPUT);
    pinMode(selPin1, OUTPUT);
    pinMode(selPin2, OUTPUT);
    pinMode(selPin3, OUTPUT);
    pinMode(selPin4, OUTPUT);

    pinMode(echoPin, INPUT);
    pinMode(trigPin, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    pack = "";
    for (int i = 0; i < 3; ++i)
    {
        select(i);
        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);
        duration = pulseIn(echoPin, HIGH);
        delayMicroseconds(10);
        distance = duration * 0.034 / 2;
        pack += String(distance) + " ";
    }
    Serial.println(pack);
}

void select(int x)
{
    for (int i = 0; i < 5; i++)
        digitalWrite(sels[i], (x >> i) & 1);
}