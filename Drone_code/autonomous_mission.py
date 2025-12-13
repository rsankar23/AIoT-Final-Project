import time
import threading
import requests

from picamera2 import Picamera2

from uav_pid_controller import UAVController, now_s, clamp


# -----------------------------
# Config
# -----------------------------
CLOUD_DETECT_URL = "https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/detect"

RUN_SECONDS = 20.0

CONTROL_HZ = 50.0
VISION_HZ = 1.0

CAM_W = 640
CAM_H = 480

HTTP_TIMEOUT_S = 5.0


THR_IDLE = 0.00
THR_HOVER_LIKE = 0.14


PAUSE_ON_PERSON = True
PERSON_PAUSE_SECONDS = 3.0


# -----------------------------
# Shared state between threads
# -----------------------------
shared = {
    "person_detected": False,
    "num_detections": 0,
    "latency_ms": None,
    "last_ok_t": 0.0,
    "last_err": None,
}
lock = threading.Lock()


def vision_loop(stop_evt: threading.Event):
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (CAM_W, CAM_H)}))
    cam.start()
    time.sleep(0.4)

    headers = {"Content-Type": "image/jpeg"}
    period = 1.0 / VISION_HZ

    while not stop_evt.is_set():
        t0 = time.monotonic()
        try:
            jpeg = cam.capture_file(None, format="jpeg")
            r = requests.post(
                CLOUD_DETECT_URL,
                data=jpeg,
                headers=headers,
                timeout=HTTP_TIMEOUT_S
            )
            data = r.json()

            with lock:
                shared["person_detected"] = bool(data.get("person_detected", False))
                shared["num_detections"] = int(data.get("num_detections", 0))
                shared["latency_ms"] = data.get("latency_ms", None)
                shared["last_ok_t"] = time.monotonic()
                shared["last_err"] = None
        except Exception as e:
            with lock:
                shared["last_err"] = str(e)

        dt = time.monotonic() - t0
        sleep_s = max(0.0, period - dt)
        time.sleep(sleep_s)

    cam.stop()
    cam.close()


def mission_setpoints(ctrl: UAVController, t_elapsed: float, person_hold: bool):
    """
    Predetermined path profile (time-based).
    This does NOT require GPS/SLAM.
    Keeps roll/pitch near 0 (level), optionally changes yaw setpoint.
    """
    ctrl.set_attitude_setpoints(roll_deg=0.0, pitch_deg=0.0)

    if person_hold:
        ctrl.set_throttle(THR_HOVER_LIKE)
        return

    phase = t_elapsed % 12.0

    if phase < 2.0:
        ctrl.set_throttle(THR_IDLE)
    elif phase < 6.0:
        ctrl.set_throttle(THR_HOVER_LIKE)
    elif phase < 8.0:
        ctrl.set_throttle(THR_HOVER_LIKE)
        ctrl.set_attitude_setpoints(roll_deg=0.0, pitch_deg=0.0, yaw_deg=ctrl.yaw_sp + 15.0)
    elif phase < 10.0:
        ctrl.set_throttle(THR_HOVER_LIKE)
        ctrl.set_attitude_setpoints(roll_deg=0.0, pitch_deg=0.0, yaw_deg=ctrl.yaw_sp - 15.0)
    else:
        ctrl.set_throttle(THR_HOVER_LIKE)


def run_control_and_demo():
    ctrl = UAVController()
    ctrl.initialize()

    time.sleep(1.0)
    ctrl.arm()

    stop_evt = threading.Event()
    vt = threading.Thread(target=vision_loop, args=(stop_evt,), daemon=True)
    vt.start()

    dt = 1.0 / CONTROL_HZ
    t_next = now_s()

    t_start = now_s()
    t_end = t_start + RUN_SECONDS

    person_pause_until = 0.0
    last_log_t = 0.0

    while now_s() < t_end:
        t = now_s()
        if t < t_next:
            time.sleep(max(0.0, t_next - t))
            continue
        t_next += dt

        with lock:
            person = shared["person_detected"]
            nd = shared["num_detections"]
            lat = shared["latency_ms"]
            err = shared["last_err"]

        if PAUSE_ON_PERSON and person and t > person_pause_until:
            person_pause_until = t + PERSON_PAUSE_SECONDS

        person_hold = t < person_pause_until

        mission_setpoints(ctrl, t - t_start, person_hold)

        ctrl.step(t)

        if t - last_log_t > 1.0:
            last_log_t = t
            print(
                f"t={t - t_start:5.1f}s "
                f"thr={ctrl.throttle:0.2f} "
                f"person={person} dets={nd} lat_ms={lat} "
                f"{'HOLD' if person_hold else 'RUN'} "
                f"{'' if err is None else ('err=' + err)}"
            )

    stop_evt.set()
    time.sleep(0.2)

    ctrl.kill()
    time.sleep(0.5)


def main():
    ctrl = None
    try:
        run_control_and_demo()
    except Exception as e:
        print("Fatal error:", e)
        try:
            if ctrl is not None:
                ctrl.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()