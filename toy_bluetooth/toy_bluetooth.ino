#include <SoftwareSerial.h> // 引入软件串口库

SoftwareSerial bluetooth(2, 3); // 定义软件串口对象，将蓝牙模块的TX和RX引脚分别连接到Arduino的数字引脚2和3

void setup() {
  Serial.begin(9600); // 初始化硬件串口，用于与电脑进行通信
  bluetooth.begin(9600); // 初始化软件串口，用于与蓝牙模块进行通信
}

void loop() {
  if (bluetooth.available()) { // 如果蓝牙模块有数据发送过来
    char c = bluetooth.read(); // 读取接收到的字符
    Serial.print(c); // 通过硬件串口打印接收到的字符
  }
}
