// #include <Arduino.h>
#include <HardwareSerial.h>
#include "Freenove_WS2812_Lib_for_ESP32.h"

#define LEDS_COUNT  3
#define LEDS_PIN	48
#define CHANNEL		0
#define SERIAL_BAUD 9600
#define DEBUGER_BAUD 9600
#define RXD1 18
#define TXD1 17

HardwareSerial cardSerial(1); // use UART1
Freenove_ESP32_WS2812 strip = Freenove_ESP32_WS2812(LEDS_COUNT, LEDS_PIN, CHANNEL);

char number  = 'x';
int incomingByte = 2;

void lightRGB(char r, char g, char b) {
  for (int i = 0; i < LEDS_COUNT; i++) {
    strip.setLedColorData(i, r, g, b);
  }
  strip.show();
}

void setup()
{
  strip.begin();
  strip.setBrightness(20);  

  lightRGB(255, 0, 255);

  Serial.begin(DEBUGER_BAUD);
  cardSerial.begin(SERIAL_BAUD , SERIAL_8N1, RXD1, TXD1);
  // cardSerial.setRxBufferSize(1);
  Serial.println("\nInit debug");
}
void loop()
{ 
  if (cardSerial.available() > 0)
  {
    String receivedData = cardSerial.readStringUntil('\n');
    Serial.println(receivedData);
    
    if (receivedData.toInt() == 0) {
      lightRGB(0, 255, 0);
    }
    else if (receivedData.toInt() == 1) {
      lightRGB(255, 0, 0);
    }
    else {
      lightRGB(128, 128, 128);
    }
  }
}