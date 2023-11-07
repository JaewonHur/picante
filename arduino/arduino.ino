//                  ㅅ,ㅌ,ㅋ,ㅍ,ㅎ, ㅊ,ㄹ,ㄴ,ㄷ,ㅂ, ㄱ,ㅇ,ㅁ,ㅈ
const int pin[14] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

void setup() {
  // put your setup code here, to run once:
  int i = 0;
  for (i = 0; i < 14; i++) {
    pinMode(pin[i], OUTPUT);
  }

  Serial.begin(9600);
}


void loop() {
  // put your main code here, to run repeatedly:
  int input[14] = {0,};
  
  if (Serial.available()) {
    while(Serial.available()) {
      int x = Serial.read();
      if (x == 255) {
        break;
      } else if (x < 14) {
        input[x] = 1;
      } else {
        continue;
      }
    }

    int i = 0;
    for (i = 0; i < 14; i++) {
      if (input[i] == 1) {
        digitalWrite(i, HIGH);
      } else {
        digitalWrite(i, LOW);
      }
    }
  }

  delay(100);
}
