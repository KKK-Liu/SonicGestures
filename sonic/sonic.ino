#define trigPin0 2
#define trigPin1 3
#define trigPin2 4
#define trigPin3 5
#define trigPin4 6

#define enable 7
#define echoPin 8

#define enableTrue false
#define enableFalse true

int sonic_pos;
long duration;
int distance;

int trigs[5] = {
    trigPin0,
    trigPin1,
    trigPin2,
    trigPin3,
    trigPin4,
};
bool ishigh[5];
String pack;

void setup(){
    pinMode(trigPin0, OUTPUT);
    pinMode(trigPin1, OUTPUT);
    pinMode(trigPin2, OUTPUT);
    pinMode(trigPin3, OUTPUT);
    pinMode(trigPin4, OUTPUT);

    pinMode(echoPin, INPUT);
    Serial.begin(9600);
}

void loop(){
    pack = "";
    for (sonic_pos = 0; sonic_pos <= 31;sonic_pos++)
    {
        output(sonic_pos);
        duration = pulseIn(echoPin, HIGH);
        distance = duration * 0.034 / 2;
        pack += String(distance) + " ";
    }
    Serial.println(pack);
}

void output(int x){
    for (int i = 0; i < 5; i++)
        ishigh[i] = (x >> i) & 1;

    // digitalWrite(enable, enableFalse);
    for (int i = 0; i < 5;i++)
        digitalWrite(trigs[i], LOW);
    // digitalWrite(enable, enableTrue);

    // delayMicroseconds(2);

    // digitalWrite(enable, enableFalse);
    delayMicroseconds(2);

    for (int i = 0; i < 5;i++)
        digitalWrite(trigs[i], ishigh[i]);
    // digitalWrite(enable, enableTrue);


    delayMicroseconds(10);

    // digitalWrite(enable, enableFalse);
    for (int i = 0; i < 5;i++)
        digitalWrite(trigs[i], LOW);
    // digitalWrite(enable, enableTrue);
}   