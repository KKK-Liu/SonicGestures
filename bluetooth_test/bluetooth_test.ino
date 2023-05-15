#include <SoftwareSerial.h>

SoftwareSerial bluetooth(2, 3); // 用于与HC-06通信的软件串口对象

void setup() {
  Serial.begin(9600); // 初始化与计算机的串口通信
  bluetooth.begin(115200); // 初始化与HC-06的串口通信
}

void loop() {
  if (bluetooth.available()) {
    char data = bluetooth.read();
    Serial.write(data); // 将接收到的数据发送到计算机
  }
  if (Serial.available()) {
    char data = Serial.read();
    bluetooth.write(data); // 将接收到的数据发送到HC-06
  }
}

// void loop() {
//   bluetooth.write()
// }
