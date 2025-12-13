import time
import board
import busio
import digitalio
import adafruit_vl53l0x

XSHUT_PINS = [
    board.D17,
    board.D22,
    board.D23,
    board.D24,
    board.D27
]

NEW_I2C_ADDRS = [
    0x30,
    0x31,
    0x32,
    0x33,
    0x34
]

i2c = busio.I2C(board.SCL, board.SDA)

xshuts = []
for pin in XSHUT_PINS:
    x = digitalio.DigitalInOut(pin)
    x.direction = digitalio.Direction.OUTPUT
    x.value = False
    xshuts.append(x)

time.sleep(0.1)

sensors = []

for i in range(len(xshuts)):
    xshuts[i].value = True
    time.sleep(0.2)

    sensor = adafruit_vl53l0x.VL53L0X(i2c)
    sensor.set_address(NEW_I2C_ADDRS[i])
    time.sleep(0.05)

    sensor = adafruit_vl53l0x.VL53L0X(i2c, address=NEW_I2C_ADDRS[i])
    sensors.append(sensor)

while True:
    for i, sensor in enumerate(sensors):
        try:
            print(f"LiDAR {i+1}: {sensor.range} mm", end=" | ")
        except RuntimeError:
            print(f"LiDAR {i+1}: out of range", end=" | ")
    print()
    time.sleep(0.5)