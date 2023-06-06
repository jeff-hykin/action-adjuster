class RcCarMotors:
    def __init__(self, *, front_left_wheel_speed_pin=None, front_right_wheel_speed_pin=None, back_left_wheel_speed_pin=None, back_right_wheel_speed_pin=None, front_left_wheel_direction_pin=None, front_right_wheel_direction_pin=None, back_left_wheel_direction_pin=None, back_right_wheel_direction_pin=None, frequency=1000):
        """
            Arguments:
                frequency: integer, units=Hertz
                front_left_wheel_speed_pin:  non-negative int, see pinout below
                front_right_wheel_speed_pin: non-negative int, see pinout below
                back_left_wheel_speed_pin:   non-negative int, see pinout below
                back_right_wheel_speed_pin:  non-negative int, see pinout below
            
            Raspberry Pi Pinout:
                
                If this is the board (Raspberry Pi 3 B)
                    ,--------------------------------.
                    | oooooooooooooooooooo J8     +====
                    | 1ooooooooooooooooooo        | USB
                    |                             +====
                    |      Pi Model 3B  V1.2         |
                    |      +----+                 +====
                    | |D|  |SoC |                 | USB
                    | |S|  |    |                 +====
                    | |I|  +----+                    |
                    |                   |C|     +======
                    |                   |S|     |   Net
                    | pwr        |HDMI| |I||A|  +======
                    `-| |--------|    |----|V|-------'

                    # Revision           : a22082
                    # SoC                : BCM2837
                    # RAM                : 1024Mb
                    # Storage            : MicroSD
                    # USB ports          : 4 (excluding power)
                    # Ethernet ports     : 1
                    # Wi-fi              : True
                    # Bluetooth          : True
                    # Camera ports (CSI) : 1
                    # Display ports (DSI): 1
                
                Then here's the pins:
                       3V3  (1) (2)  5V    
                     GPIO2  (3) (4)  5V    
                     GPIO3  (5) (6)  GND   
                     GPIO4  (7) (8)  GPIO14
                       GND  (9) (10) GPIO15
                    GPIO17 (11) (12) GPIO18
                    GPIO27 (13) (14) GND   
                    GPIO22 (15) (16) GPIO23
                       3V3 (17) (18) GPIO24
                    GPIO10 (19) (20) GND   
                     GPIO9 (21) (22) GPIO25
                    GPIO11 (23) (24) GPIO8 
                       GND (25) (26) GPIO7 
                     GPIO0 (27) (28) GPIO1 
                     GPIO5 (29) (30) GND   
                     GPIO6 (31) (32) GPIO12
                    GPIO13 (33) (34) GND   
                    GPIO19 (35) (36) GPIO16
                    GPIO26 (37) (38) GPIO20
                       GND (39) (40) GPIO21
        """
        self.front_left_wheel_speed_pin  = front_left_wheel_speed_pin
        self.front_right_wheel_speed_pin = front_right_wheel_speed_pin
        self.back_left_wheel_speed_pin   = back_left_wheel_speed_pin
        self.back_right_wheel_speed_pin  = back_right_wheel_speed_pin
        self.front_left_wheel_direction_pin  = front_left_wheel_direction_pin
        self.front_right_wheel_direction_pin = front_right_wheel_direction_pin
        self.back_left_wheel_direction_pin   = back_left_wheel_direction_pin
        self.back_right_wheel_direction_pin  = back_right_wheel_direction_pin
        self._frequency = frequency # at least I think this is frequency, help(GPIO.PWM) isn't very helpful
        
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        self.pwm_objects = []
        if self.front_left_wheel_speed_pin  != None: GPIO.setup(self.front_left_wheel_speed_pin , GPIO.OUT); self.GPIO.output(self.front_left_wheel_speed_pin , GPIO.LOW); self.front_left_wheel_pwm  = GPIO.PWM(self.front_left_wheel_speed_pin , frequency);self.pwm_objects.append(self.front_left_wheel_pwm)
        if self.front_right_wheel_speed_pin != None: GPIO.setup(self.front_right_wheel_speed_pin, GPIO.OUT); self.GPIO.output(self.front_right_wheel_speed_pin, GPIO.LOW); self.front_right_wheel_pwm = GPIO.PWM(self.front_right_wheel_speed_pin, frequency);self.pwm_objects.append(self.front_right_wheel_pwm)
        if self.back_left_wheel_speed_pin   != None: GPIO.setup(self.back_left_wheel_speed_pin  , GPIO.OUT); self.GPIO.output(self.back_left_wheel_speed_pin  , GPIO.LOW); self.back_left_wheel_pwm   = GPIO.PWM(self.back_left_wheel_speed_pin  , frequency);self.pwm_objects.append(self.back_left_wheel_pwm)
        if self.back_right_wheel_speed_pin  != None: GPIO.setup(self.back_right_wheel_speed_pin , GPIO.OUT); self.GPIO.output(self.back_right_wheel_speed_pin , GPIO.LOW); self.back_right_wheel_pwm  = GPIO.PWM(self.back_right_wheel_speed_pin , frequency);self.pwm_objects.append(self.back_right_wheel_pwm)
        self.pwm_objects = tuple(self.pwm_objects)
        if self.front_left_wheel_direction_pin  != None: GPIO.setup(self.front_left_wheel_direction_pin , GPIO.OUT); self.GPIO.output(self.front_left_wheel_direction_pin , GPIO.LOW)
        if self.front_right_wheel_direction_pin != None: GPIO.setup(self.front_right_wheel_direction_pin, GPIO.OUT); self.GPIO.output(self.front_right_wheel_direction_pin, GPIO.LOW)
        if self.back_left_wheel_direction_pin   != None: GPIO.setup(self.back_left_wheel_direction_pin  , GPIO.OUT); self.GPIO.output(self.back_left_wheel_direction_pin  , GPIO.LOW)
        if self.back_right_wheel_direction_pin  != None: GPIO.setup(self.back_right_wheel_direction_pin , GPIO.OUT); self.GPIO.output(self.back_right_wheel_direction_pin , GPIO.LOW)
        
        for each_pwm in self.pwm_objects:
            each_pwm.stop()
            each_pwm.start(0)
        
    def __del__(self):
        for each_pwm in self.pwm_objects:
            each_pwm.stop()
        self.GPIO.cleanup()
    
    # 
    # tests
    # 
    def sanity_test(self, interval_duration_in_seconds=1):
        import time
        
        print("testing:")
        if self.front_left_wheel_speed_pin  != None: print(f"    front_left_wheel : forwards 50% speed for {interval_duration_in_seconds}sec");self.front_left_wheel_velocity  = 50;time.sleep(interval_duration_in_seconds);print(f"    front_left_wheel : backwards 50% speed for {interval_duration_in_seconds}sec");self.front_left_wheel_velocity  = -50;time.sleep(interval_duration_in_seconds)
        if self.front_right_wheel_speed_pin != None: print(f"    front_right_wheel: forwards 50% speed for {interval_duration_in_seconds}sec");self.front_right_wheel_velocity = 50;time.sleep(interval_duration_in_seconds);print(f"    front_right_wheel: backwards 50% speed for {interval_duration_in_seconds}sec");self.front_right_wheel_velocity = -50;time.sleep(interval_duration_in_seconds)
        if self.back_left_wheel_speed_pin   != None: print(f"    back_left_wheel  : forwards 50% speed for {interval_duration_in_seconds}sec");self.back_left_wheel_velocity   = 50;time.sleep(interval_duration_in_seconds);print(f"    back_left_wheel  : backwards 50% speed for {interval_duration_in_seconds}sec");self.back_left_wheel_velocity   = -50;time.sleep(interval_duration_in_seconds)
        if self.back_right_wheel_speed_pin  != None: print(f"    back_right_wheel : forwards 50% speed for {interval_duration_in_seconds}sec");self.back_right_wheel_velocity  = 50;time.sleep(interval_duration_in_seconds);print(f"    back_right_wheel : backwards 50% speed for {interval_duration_in_seconds}sec");self.back_right_wheel_velocity  = -50;time.sleep(interval_duration_in_seconds)
        print("testing complete")
    
    
    # 
    # actions
    # 
    def full_stop(self):
        for each_pwm in self.pwm_objects:
            each_pwm.ChangeDutyCycle(0)
    
    # 
    # wheels
    # 
    def _velocity_to_direction_and_speed(self, velocity):
        direction = self.GPIO.HIGH if velocity >= 0 else self.GPIO.LOW
        speed = abs(round(velocity))
        return direction, speed
        
    @property
    def front_left_wheel_velocity(self): return self._front_left_wheel_velocity
    @front_left_wheel_velocity.setter
    def front_left_wheel_velocity(self, value):
        """
            Arguments: 
                value should be between -100 and 100
                100 being forward at max velocity
                -100 being backwards at max velocity
        """
        self._front_left_wheel_velocity = value
        direction, speed = self._velocity_to_direction_and_speed(value)
        self.GPIO.output(self.front_left_wheel_direction_pin, direction)
        self.front_left_wheel_pwm.ChangeDutyCycle(speed)
    
    @property
    def front_right_wheel_velocity(self): return self._front_right_wheel_velocity
    @front_right_wheel_velocity.setter
    def front_right_wheel_velocity(self, value):
        """
            Arguments: 
                value should be between -100 and 100
                100 being forward at max velocity
                -100 being backwards at max velocity
        """
        self._front_right_wheel_velocity = value
        direction, speed = self._velocity_to_direction_and_speed(value)
        self.GPIO.output(self.front_right_wheel_direction_pin, direction)
        self.front_right_wheel_pwm.ChangeDutyCycle(speed)
    
    @property
    def back_left_wheel_velocity(self): return self._back_left_wheel_velocity
    @back_left_wheel_velocity.setter
    def back_left_wheel_velocity(self, value):
        """
            Arguments: 
                value should be between -100 and 100
                100 being forward at max velocity
                -100 being backwards at max velocity
        """
        self._back_left_wheel_velocity = value
        direction, speed = self._velocity_to_direction_and_speed(value)
        self.GPIO.output(self.back_left_wheel_direction_pin, direction)
        self.back_left_wheel_pwm.ChangeDutyCycle(speed)
    
    @property
    def back_right_wheel_velocity(self): return self._back_right_wheel_velocity
    @back_right_wheel_velocity.setter
    def back_right_wheel_velocity(self, value):
        """
            Arguments: 
                value should be between -100 and 100
                100 being forward at max velocity
                -100 being backwards at max velocity
        """
        self._back_right_wheel_velocity = value
        direction, speed = self._velocity_to_direction_and_speed(value)
        self.GPIO.output(self.back_right_wheel_direction_pin, direction)
        self.back_right_wheel_pwm.ChangeDutyCycle(speed)
        
    # 
    # frequency getter/setter
    # 
    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        # change the frequency of all the pinout options
        for each in self.pwm_objects:
            each.ChangeFrequency(self._frequency)

car = RcCarMotors(
    back_right_wheel_direction_pin=31, # GPIO6
    back_right_wheel_speed_pin=33,     # GPIO13
    back_left_wheel_direction_pin=35,  # GPIO19
    back_left_wheel_speed_pin=37,      # GPIO26
)