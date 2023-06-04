void setup() {
  Serial.begin(9600);               // initialize serial communication at 9600 bits per second:
  Serial1.begin(9600);
}
void loop() {
  Serial1.println('1');
  delay(2000);
  Serial1.println('0');
  delay(2000);
}