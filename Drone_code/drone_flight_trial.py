import time
from uav_pid_controller import UAVController, now_s, clamp


def run_for(ctrl, seconds, hz=50.0):
    dt = 1.0 / hz
    t_end = now_s() + seconds
    t_next = now_s()

    while now_s() < t_end:
        t = now_s()
        if t < t_next:
            time.sleep(max(0.0, t_next - t))
            continue
        t_next += dt
        ctrl.step(t)


def main():
    ctrl = UAVController()
    ctrl.initialize()

    time.sleep(1.0)
    ctrl.arm()

    ctrl.set_attitude_setpoints(roll_deg=0.0, pitch_deg=0.0)

    ctrl.set_throttle(0.00)
    run_for(ctrl, 2.0)

    for thr in [0.05, 0.08, 0.10, 0.12, 0.14]:
        ctrl.set_throttle(clamp(thr, 0.0, 1.0))
        run_for(ctrl, 1.0)

    ctrl.set_throttle(0.14)
    run_for(ctrl, 3.0)

    for thr in [0.12, 0.10, 0.08, 0.05, 0.00]:
        ctrl.set_throttle(clamp(thr, 0.0, 1.0))
        run_for(ctrl, 1.0)

    ctrl.kill()
    time.sleep(0.5)


if __name__ == "__main__":
    main()