# Drone Control Software – PID Controller & Mission Logic

## Overview
This repository contains the drone-side control software developed to support a mobile deployment of the vision-based assistive detection system. The software is structured to demonstrate how onboard sensing, closed-loop control, and mission-level logic can be integrated on a Raspberry Pi–based aerial platform.

The implementation focuses on:
- A PID-based stabilization and control framework
- Sensor integration via I²C (IMU and distance sensors)
- A time-based mission script for deterministic flight behavior
- Hardware abstraction for motor control using a PWM controller

While full autonomous flight was not demonstrated within the project timeframe, the software establishes the core control architecture required for closed-loop stabilization and extensible mission execution.

---

## Software Architecture
The dautonomous_mission.py is divided into two primary layers:

1. **Low-Level Control Layer (PID Controller)**
2. **High-Level Mission Layer (Mission Script)**

This separation ensures that time-critical stabilization logic is isolated from mission sequencing and higher-level behaviors.

---

## PID Controller Module

### Purpose
The PID controller module is responsible for stabilizing the drone by continuously correcting deviations between desired setpoints and measured states. It is designed to operate at a high update rate and provide smooth, stable motor commands.

### Inputs
- Orientation and motion data from the **Adafruit BNO055 IMU**
  - Roll
  - Pitch
  - Yaw
- (Planned extension) Distance measurements from **VL53L0X LiDAR sensors**

Where:
- `e(t)` is the error between a desired setpoint and the measured state
- `Kp`, `Ki`, and `Kd` are tunable gains

Separate PID loops are intended for:
- **Attitude control** (roll, pitch, yaw)
- **Altitude control** (vertical thrust)
- **Position control** (outer-loop planning, future extension)

### Output
The PID controller produces normalized thrust and torque commands, which are translated into PWM duty cycles and sent to the motor controllers via the **PCA9685 PWM HAT**.

---

## Motor Control and PWM Interface

### PCA9685 PWM HAT
Motor outputs are generated using an Adafruit PCA9685 16-channel PWM controller over I²C. This design offloads PWM timing from the Raspberry Pi CPU, ensuring stable, jitter-free motor signals even while the processor is handling camera capture and network communication.

Each motor (or thruster) is assigned a dedicated PWM channel, and throttle values are updated continuously by the PID controller.

---

## Mission Script

### Purpose
The mission script defines a deterministic, time-based flight sequence intended for demonstration and testing. Rather than relying on fully autonomous navigation, the mission logic executes a predefined set of actions while the PID controller maintains stability.

### Example Mission Profile
A typical mission sequence includes:
1. **Initialization**
   - Sensor warm-up
   - PID controller reset
   - Motor arming
2. **Takeoff / Vertical Thrust**
   - Apply controlled upward thrust for a fixed duration
3. **Hover Phase**
   - Maintain stable attitude using PID control
4. **Yaw or Translation Segment**
   - Apply differential thrust to induce rotation or forward motion
5. **Landing / Shutdown**
   - Gradual reduction of thrust
   - Safe motor stop

The mission script invokes control functions exposed by the PID controller module, allowing the same control logic to be reused across different mission profiles.

---

## Interaction Between Mission and Control Layers
- The **mission script** sets high-level targets (e.g., desired yaw rate, thrust level).
- The **PID controller** continuously adjusts motor outputs to track these targets.
- Sensor feedback is processed in real time to close the control loop.
- Control execution is time-driven and deterministic, making it suitable for scripted demonstrations.

---

## Safety and Design Considerations
- Control outputs are clamped to safe throttle limits.
- Integral windup protection is included in the PID design.
- PWM generation is hardware-timed to avoid jitter.
- The system is designed to fail safely by stopping motors if control execution is interrupted.

---

## Limitations and Future Extensions
- Full autonomous navigation and obstacle avoidance were not completed within the project timeframe.
- Planned extensions include:
  - Vision-guided navigation using cloud inference results
  - Adaptive PID gain tuning

---

## Summary
This drone control software demonstrates a complete control-stack design for a small aerial platform, including sensor integration, PID-based stabilization, and mission-level execution. The implementation establishes a robust and extensible foundation for future work in mobile, vision-assisted robotic systems.

The main code is Autonomous_mission.py which imports the PID_controller.py code as well as sensor libraries, to fly the drone in a pre-determined path while taking images at 1-5hz and sending it to the GCP for inference.