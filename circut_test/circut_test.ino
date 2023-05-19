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

int sels[5] = {
    selPin0,
    selPin1,
    selPin2,
    selPin3,
    selPin4,
};

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
    // String num_s = Serial.readString();

    // int num = num_s.toInt();
    select(8);
    delay(5000);
    for (int i = 0; i < 5; i++)
        digitalWrite(sels[i], 0);
}

void select(int x)
{
    for (int i = 0; i < 5; i++)
        digitalWrite(sels[i], (x >> i) & 1);
}