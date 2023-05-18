#define selPin0 2
#define selPin1 3
#define selPin2 4
#define selPin3 5
#define selPin4 6

#define trigPin1 7
#define trigPin2 8
#define trigPin3 9
#define trigPin4 10
#define echoPin1 11
#define echoPin2 12
#define echoPin3 13
#define echoPin4 14

#define enableTrue false
#define enableFalse true

int sonic_pos;
long duration;
int distance;
int trigPin = trigPin1;
int echoPin = echoPin1;

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

    pinMode(echoPin1, INPUT);
    pinMode(echoPin2, INPUT);
    pinMode(echoPin3, INPUT);
    pinMode(echoPin4, INPUT);

    pinMode(trigPin1, OUTPUT);
    pinMode(trigPin2, OUTPUT);
    pinMode(trigPin3, OUTPUT);
    pinMode(trigPin4, OUTPUT);
    Serial.begin(115200);
}

void loop()
{
    pack = "";
    for (int i = 0; i < 24; ++i)
    {
        select(i);
        delay(2);
        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);
        duration = pulseIn(echoPin, HIGH);
        delayMicroseconds(10);
        distance = int(duration * 0.034 / 2);
        pack += String(distance) + " ";
    }
    Serial.println(pack);
}

void select(int x)
{
    for (int i = 0; i < 5; i++)
        digitalWrite(sels[i], (x >> i) & 1);
    if (x < 6)
    {
        echoPin = echoPin1;
        trigPin = trigPin1;
    }
    else if (x < 12)
    {
        echoPin = echoPin2;
        trigPin = trigPin2;
    }
    else if (x < 18)
    {
        echoPin = echoPin3;
        trigPin = trigPin3;
    }
    else if (x < 24)
    {
        echoPin = echoPin4;
        trigPin = trigPin4;
    }
}