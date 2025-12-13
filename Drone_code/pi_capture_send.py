import time
import requests
from picamera2 import Picamera2

API_URL = "https://uav-rescue-detection-xhp2lso5pq-uc.a.run.app/detect"
TIMEOUT_S = 30

cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(0.4)

jpeg = cam.capture_file(None, format="jpeg")

cam.stop()
cam.close()

headers = {"Content-Type": "image/jpeg"}
r = requests.post(API_URL, data=jpeg, headers=headers, timeout=TIMEOUT_S)

print("HTTP", r.status_code)
print(r.json())