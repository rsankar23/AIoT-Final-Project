import time
import math

import board
import busio
import digitalio

import adafruit_bno055
import adafruit_vl53l0x
from adafruit_pca9685 import PCA9685


# -----------------------------
# Utility helpers
# -----------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_degrees(angle_deg):
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def now_s():
    return time.monotonic()


# -----------------------------
# PID Controller
# -----------------------------

class PID:
    """
    Basic PID controller with:
      - derivative on measurement (less noise than derivative on error)
      - integrator clamping (anti-windup)
      - output clamping
    """

    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0, i_min=-0.5, i_max=0.5):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.i_min = float(i_min)
        self.i_max = float(i_max)

        self.integral = 0.0
        self.prev_meas = None
        self.prev_t = None

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None
        self.prev_t = None

    def update(self, setpoint, measurement, t):
        if self.prev_t is None:
            self.prev_t = t
            self.prev_meas = measurement
            return 0.0

        dt = t - self.prev_t
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement

        self.integral += error * dt
        self.integral = clamp(self.integral, self.i_min, self.i_max)

        d_meas = (measurement - self.prev_meas) / dt
        d_term = -self.kd * d_meas

        p_term = self.kp * error
        i_term = self.ki * self.integral

        out = p_term + i_term + d_term
        out = clamp(out, self.out_min, self.out_max)

        self.prev_t = t
        self.prev_meas = measurement
        return out


# -----------------------------
# Sensors
# -----------------------------

class BNO055IMU:
    def __init__(self, i2c):
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def read_euler_deg(self):
        e = self.sensor.euler
        if e is None:
            return None
        heading, roll, pitch = e  # Adafruit order: heading, roll, pitch
        if heading is None or roll is None or pitch is None:
            return None
        return float(roll), float(pitch), float(heading)


class MultiVL53L0X:
    """
    5x VL53L0X on one I2C bus using XSHUT + runtime address assignment.
    """

    def __init__(self, i2c, xshut_pins, addrs):
        if len(xshut_pins) != len(addrs):
            raise ValueError("xshut_pins and addrs must have same length")
        self.i2c = i2c
        self.xshut_pins = xshut_pins
        self.addrs = addrs

        self.xshuts = []
        self.sensors = []

    def init_sensors(self):
        self.xshuts = []
        for pin in self.xshut_pins:
            x = digitalio.DigitalInOut(pin)
            x.direction = digitalio.Direction.OUTPUT
            x.value = False
            self.xshuts.append(x)

        time.sleep(0.1)

        self.sensors = []
        for i in range(len(self.xshuts)):
            self.xshuts[i].value = True
            time.sleep(0.2)

            s = adafruit_vl53l0x.VL53L0X(self.i2c)  # default 0x29
            s.set_address(self.addrs[i])
            time.sleep(0.05)

            s = adafruit_vl53l0x.VL53L0X(self.i2c, address=self.addrs[i])
            self.sensors.append(s)

    def read_mm(self):
        out = []
        for s in self.sensors:
            try:
                out.append(int(s.range))
            except RuntimeError:
                out.append(None)
        return out


# -----------------------------
# Motor control via PCA9685
# -----------------------------

class MotorMixerQuadX:
    """
    Quad-X mixer (typical). Output order: M0, M1, M2, M3.
    You MUST verify motor mapping and signs for your frame.
    """

    def mix(self, throttle, roll_cmd, pitch_cmd, yaw_cmd):
        m0 = throttle + pitch_cmd + roll_cmd - yaw_cmd
        m1 = throttle + pitch_cmd - roll_cmd + yaw_cmd
        m2 = throttle - pitch_cmd - roll_cmd - yaw_cmd
        m3 = throttle - pitch_cmd + roll_cmd + yaw_cmd
        return [m0, m1, m2, m3]


class PCA9685ESC:
    """
    Drive ESCs using PCA9685 duty_cycle.

    Two common approaches:
      (A) Use servo pulse widths (1000–2000us) and map to duty cycle.
      (B) Use normalized duty and tune experimentally.

    This class uses pulse-width mapping because it is report-friendly and standard.
    """

    def __init__(self, i2c, freq_hz=50):
        self.pca = PCA9685(i2c)
        self.pca.frequency = freq_hz

        self.freq_hz = freq_hz
        self.period_us = 1_000_000.0 / float(freq_hz)

       
        self.min_us = 1000.0
        self.max_us = 2000.0

        self.motor_channels = [0, 1, 2, 3]

    def set_pulse_range_us(self, min_us, max_us):
        self.min_us = float(min_us)
        self.max_us = float(max_us)

    def _us_to_duty16(self, pulse_us):
        pulse_us = clamp(pulse_us, 0.0, self.period_us)
        duty_frac = pulse_us / self.period_us
        return int(clamp(duty_frac, 0.0, 1.0) * 65535.0)

    def write_motor_norm(self, motor_index, value_norm):
        """
        value_norm in [0..1] maps to [min_us..max_us].
        """
        v = clamp(value_norm, 0.0, 1.0)
        pulse_us = self.min_us + v * (self.max_us - self.min_us)
        duty16 = self._us_to_duty16(pulse_us)
        ch = self.motor_channels[motor_index]
        self.pca.channels[ch].duty_cycle = duty16

    def write_all_motors_norm(self, values_norm):
        for i, v in enumerate(values_norm):
            self.write_motor_norm(i, v)

    def stop_all(self):
        for i in range(4):
            self.write_motor_norm(i, 0.0)


# -----------------------------
# High-level controller
# -----------------------------

class UAVController:
    """
    High-level control loop:
      - reads IMU
      - reads LiDARs (optional)
      - computes attitude PID outputs
      - mixes to motor commands
      - outputs PWM via PCA9685
    """

    def __init__(self):
        self.i2c = board.I2C()

        self.imu = BNO055IMU(self.i2c)

        self.lidar = MultiVL53L0X(
            self.i2c,
            xshut_pins=[board.D17, board.D22, board.D23, board.D24, board.D27],
            addrs=[0x30, 0x31, 0x32, 0x33, 0x34],
        )

        self.esc = PCA9685ESC(self.i2c, freq_hz=50)  # common ESC freq
        self.mixer = MotorMixerQuadX()

        # Attitude setpoints (degrees)
        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.yaw_sp = 0.0  

        self.throttle = 0.0


        self.pid_roll = PID(kp=0.015, ki=0.005, kd=0.0015, out_min=-0.25, out_max=0.25, i_min=-0.15, i_max=0.15)
        self.pid_pitch = PID(kp=0.015, ki=0.005, kd=0.0015, out_min=-0.25, out_max=0.25, i_min=-0.15, i_max=0.15)
        self.pid_yaw = PID(kp=0.010, ki=0.003, kd=0.0008, out_min=-0.20, out_max=0.20, i_min=-0.10, i_max=0.10)


        self.enable_lidar_avoid = True
        self.wall_mm = 450  
        self.avoid_gain_deg = 8.0  # how much tilt bias to add

        self.armed = False
        self.killed = False

    def initialize(self):
        self.lidar.init_sensors()
        self.esc.stop_all()
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()

        e = self.imu.read_euler_deg()
        if e is not None:
            r, p, y = e
            self.yaw_sp = y  # capture heading for yaw hold

    def arm(self):
        self.killed = False
        self.armed = True
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()

        e = self.imu.read_euler_deg()
        if e is not None:
            r, p, y = e
            self.yaw_sp = y

    def kill(self):
        self.killed = True
        self.armed = False
        self.throttle = 0.0
        self.esc.stop_all()

    def set_throttle(self, value_norm):
        self.throttle = clamp(value_norm, 0.0, 1.0)

    def set_attitude_setpoints(self, roll_deg, pitch_deg, yaw_deg=None):
        self.roll_sp = float(roll_deg)
        self.pitch_sp = float(pitch_deg)
        if yaw_deg is not None:
            self.yaw_sp = float(yaw_deg)

    def _lidar_avoid_bias(self, lidar_mm_list):
        """
        Example mapping (you should adapt to your sensor placement):
          Suppose sensors are arranged around the drone:
            idx0: front, idx1: front-left, idx2: left, idx3: right, idx4: front-right
        This function generates a small roll/pitch bias to steer away from walls.
        """
        if not self.enable_lidar_avoid:
            return 0.0, 0.0

        if lidar_mm_list is None or len(lidar_mm_list) != 5:
            return 0.0, 0.0

        def close(mm):
            return (mm is not None) and (mm < self.wall_mm)

        pitch_bias = 0.0
        roll_bias = 0.0

        front = lidar_mm_list[0]
        front_left = lidar_mm_list[1]
        left = lidar_mm_list[2]
        right = lidar_mm_list[3]
        front_right = lidar_mm_list[4]

        if close(front):
            pitch_bias -= self.avoid_gain_deg  # pitch back away from front wall

        if close(left) or close(front_left):
            roll_bias += self.avoid_gain_deg  # roll right away from left wall

        if close(right) or close(front_right):
            roll_bias -= self.avoid_gain_deg  # roll left away from right wall

        roll_bias = clamp(roll_bias, -12.0, 12.0)
        pitch_bias = clamp(pitch_bias, -12.0, 12.0)
        return roll_bias, pitch_bias

    def step(self, t):
        if self.killed:
            self.esc.stop_all()
            return

        if not self.armed:
            self.esc.stop_all()
            return

        euler = self.imu.read_euler_deg()
        if euler is None:
            self.esc.stop_all()
            return

        roll_meas, pitch_meas, yaw_meas = euler

        lidar_mm = self.lidar.read_mm()
        roll_bias, pitch_bias = self._lidar_avoid_bias(lidar_mm)

        roll_sp = self.roll_sp + roll_bias
        pitch_sp = self.pitch_sp + pitch_bias
        yaw_err = wrap_degrees(self.yaw_sp - yaw_meas)
        yaw_sp_eff = yaw_meas + yaw_err

        roll_cmd = self.pid_roll.update(roll_sp, roll_meas, t)
        pitch_cmd = self.pid_pitch.update(pitch_sp, pitch_meas, t)
        yaw_cmd = self.pid_yaw.update(yaw_sp_eff, yaw_meas, t)

        motors = self.mixer.mix(self.throttle, roll_cmd, pitch_cmd, yaw_cmd)

        # Normalize motors to [0..1] and clamp
        motors_clamped = [clamp(m, 0.0, 1.0) for m in motors]

        self.esc.write_all_motors_norm(motors_clamped)

        # Return telemetry for logging / report screenshots
        return {
            "t": t,
            "euler_deg": (roll_meas, pitch_meas, yaw_meas),
            "setpoints_deg": (roll_sp, pitch_sp, self.yaw_sp),
            "lidar_mm": lidar_mm,
            "pid_cmd": (roll_cmd, pitch_cmd, yaw_cmd),
            "motors": motors_clamped,
        }



def main():
    ctrl = UAVController()
    ctrl.initialize()

    print("Initialized in 2 seconds...")
    time.sleep(2.0)
    ctrl.arm()

   
    ctrl.set_throttle(0.10)
    ctrl.set_attitude_setpoints(roll_deg=0.0, pitch_deg=0.0)

    hz = 50.0
    dt = 1.0 / hz
    next_t = now_s()

    print("Running control loop. Press Ctrl+C to kill.")
    try:
        while True:
            t = now_s()
            if t < next_t:
                time.sleep(max(0.0, next_t - t))
                continue
            next_t += dt

            telemetry = ctrl.step(t)
            if telemetry is not None and int(t * 2) % 2 == 0:
                r, p, y = telemetry["euler_deg"]
                m = telemetry["motors"]
                print(f"RPY=({r:+6.2f},{p:+6.2f},{y:+6.2f}) motors={m} lidar={telemetry['lidar_mm']}")

    except KeyboardInterrupt:
        print("\nKILL")
        ctrl.kill()


if __name__ == "__main__":
    main()