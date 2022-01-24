/*
  UDPSendReceive.pde:
  This sketch receives UDP message strings, prints them to the serial port
  and sends an "acknowledge" string back to the sender

  A Processing sketch is included at the end of file that can be used to send
  and received messages for testing with a computer.

  created 21 Aug 2010
  by Michael Margolis

  This code is in the public domain.

  adapted from Ethernet library examples
*/


#include <ESP8266WiFi.h>
//#include <WiFi.h>   
#include <WiFiUdp.h>
#include <Servo.h>   // to control the servo motors

#ifndef STASSID
#define STASSID "GuilleWifi"
#define STAPSK  "guillepivon"
#endif

Servo servoMotor;

unsigned int localPort = 8888;      // local port to listen on

// buffers for receiving and sending data
char packetBuffer[UDP_TX_PACKET_MAX_SIZE + 1]; //buffer to hold incoming packet,
char  ReplyBuffer[] = "Hello Python, I am ESP8266";       // a string to send back
int angle_to_move = 90;
String negative = String('-');
String positive = String('+');

struct payload_in{
  float first; 
  float second; 
  };


// MOTOR DRIVER
#define enA 14    // enable the motor A at pin D5
#define enB 2     // enable the motor B at pin D4
#define in1 16    // pin D0      
#define in2 5     // pin D1
#define in3 4     // pin D2
#define in4 0     // pin D3

// solenoid
#define SOLENOID 2                     // Defining a LED connected to the GPIO 2
bool solenoid;                         // Define the variable "led" as a bool

int velocity = 0;

WiFiUDP Udp;

void setup() {
  Serial.begin(115200);
  
  servoMotor.attach(D7);
  servoMotor.write(angle_to_move);

  pinMode(enA, OUTPUT);     // Define the pins of the motor shield as outputs
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  //pinMode(LED, OUTPUT);     // Define the LED that will act as the robot's lights
  pinMode(SOLENOID, OUTPUT);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(STASSID, STAPSK);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
  Serial.printf("UDP server on port %d\n", localPort);
  Udp.begin(localPort);
}

void loop() {
  // if there's data available, read a packet
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    
    Serial.printf("Received packet of size %d from Python %s:%d\n    (to Arduino %s:%d, free heap = %d B)\n",
                  packetSize,
                  Udp.remoteIP().toString().c_str(), Udp.remotePort(),
                  Udp.destinationIP().toString().c_str(), Udp.localPort(),
                  ESP.getFreeHeap());
    
    
    // read the packet into packetBufffer
    int n = Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
    packetBuffer[n] = 0;
    
    //payload_in = struct.unpack("ff", packetBuffer);
    //Serial.print(payload_in);
    
    Serial.println("Contents received from Python:");
    Serial.println(packetBuffer);
    
    String udp_packet_sent = packetBuffer;
    //String sign_angle = udp_packet_sent.substring(0,1); // Creating a substring from the packet sent to obtain information regarding the sign
    String sign_angle = String('-');
    String angle = udp_packet_sent.substring(5,12);
    String vel_s = udp_packet_sent.substring(1,5);

    Serial.print("Sign: ");
    Serial.println(sign_angle);
    Serial.print("Angle: ");
    Serial.println(angle);
    Serial.print("Velocity: ");
    Serial.println(vel_s);
  
    // ROBOT STEER
    if (sign_angle == negative){
      angle_to_move = 90+angle.toInt()*2;
    }
    else if (sign_angle == positive){
      angle_to_move = 90-angle.toInt()*2;
    }
    // Serial.print("Angle to be sent to servo.write: ");
    // Serial.println(angle_to_move);
    servoMotor.write(angle_to_move);

    delay(2000);
    // BALL ENTER
    
    // ROBOT SHOOT
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);

    // PWM to be sent to the enable pins (from 0 to 255)
    float vel_f = vel_s.toFloat();
    velocity = map(vel_f,4.0,8.0,100.0,255.0);
      
    analogWrite(enA, velocity); // Send PWM signal to motor A
    analogWrite(enB, velocity); // Send PWM signal to motor B

    Serial.print("Velocity to be sent in analog: ");
    Serial.println(velocity);
    
    delay(5000);
    analogWrite(enA, 0); // STOP motor A
    analogWrite(enB, 0); // STOP motor B
    
    // send a reply, to the IP address and port that sent us the packet we received
    Serial.println("Now send a reply message to Python");
    Serial.println("__________________________________");
    Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
    Udp.write(ReplyBuffer);
    Udp.endPacket();
  }

}
