#include <Servo.h>

Servo servo;

void setup() {
  Serial.begin(9600);

  servo.write(93);  // 先设置停止值
  servo.attach(2);

  Serial.println("输入 s 开始转动");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 's') {
      Serial.println("开始转动");

      servo.write(97);  
      delay(600);

      servo.write(93);  // 停止
      Serial.println("转动完成");
    }
  }
}