#include <Servo.h>

Servo servo;

void setup() {
  servo.attach(3);   // 舵机信号线接 D9
}

void loop() {
  servo.write(80);    // 一个方向全速转
  delay(2000);       // 转 2 秒

  servo.write(90);   // 停止
  delay(1000);       // 停 1 秒

  servo.write(100);  // 反方向全速转
  delay(2000);       // 转 2 秒

  servo.write(90);   // 停止
  delay(1000);
}