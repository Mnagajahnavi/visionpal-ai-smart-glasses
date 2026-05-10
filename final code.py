import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import pyttsx3
import requests
from picamera2 import Picamera2


# -------------------------------
# IMAGGA API
# -------------------------------
# 🔑 Your Imagga API credentials
API_KEY = "acc_7055d8b2de8746e"
API_SECRET = "c8704997ca8e97b2d8dcc02cc67ad3c9"

# -------------------------------------------------
# 🔊 SPEAKER
# -------------------------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print("Speak:", text)
    engine.say(text)
    engine.runAndWait()

# -------------------------------------------------
# 📏 GPIO (Ultrasonic)
# -------------------------------------------------
TRIG = 2
ECHO = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# -------------------------------------------------
# 📸 CAMERA
# -------------------------------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)})
)
picam2.start()

# -------------------------------------------------
# 🧠 YOLO
# -------------------------------------------------
weights = "yolov4-tiny.weights"
cfg = "yolov4-tiny.cfg"
labels = "labels.txt"

with open(labels, "r") as f:
    class_names = [c.strip() for c in f.readlines()]

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layerNames = net.getLayerNames()
output_layers = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

CONFIDENCE = 0.3
NMS = 0.4

# -------------------------------------------------
# 📏 DISTANCE FUNCTION
# -------------------------------------------------
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    end = time.time()

    while GPIO.input(ECHO) == 0:
        start = time.time()

    while GPIO.input(ECHO) == 1:
        end = time.time()

    distance = (end - start) * 17150
    return round(distance, 2)

# -------------------------------------------------
# 🌐 IMAGGA DESCRIPTION
# -------------------------------------------------
def imagga_describe(image_path):
    try:
        response = requests.post(
            "https://api.imagga.com/v2/tags",
            auth=(API_KEY, API_SECRET),
            files={"image": open(image_path, "rb")}
        )

        data = response.json()
        tags = data['result']['tags'][:5]

        words = [tag['tag']['en'] for tag in tags]

        # Natural sentence
        if "person" in words:
            description = "A person is in front of you"
        elif "food" in words:
            description = "There is food in front of you"
        elif "vehicle" in words:
            description = "A vehicle is detected"
        else:
            description = "This image contains " + ", ".join(words)

        print("Imagga:", description)
        speak(description)

    except Exception as e:
        print("Imagga Error:", e)
        speak("Unable to describe image")

# -------------------------------------------------
# 🟢 MAIN LOOP
# -------------------------------------------------
print("System started...")

last_spoken = ""

try:
    while True:

        frame = picam2.capture_array()

        # FIX RGBA → BGR
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        (H, W) = frame.shape[:2]

        distance = get_distance()
        print("Distance:", distance)

        # 🔥 TRIGGER CONDITION
        if distance < 20:

            speak("Object detected")

            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416),
                swapRB=True, crop=False
            )

            net.setInput(blob)
            layerOutputs = net.forward(output_layers)

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:

                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > CONFIDENCE:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (cX, cY, w, h) = box.astype("int")

                        x = int(cX - w / 2)
                        y = int(cY - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS)

            if len(idxs) > 0:
                spoken_now = set()

                for i in idxs.flatten():
                    x, y, w, h = boxes[i]
                    label = class_names[classIDs[i]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    spoken_now.add(label)

                # 🔊 Speak YOLO objects
                for obj in spoken_now:
                    if obj != last_spoken:
                        speak(obj)
                        last_spoken = obj

                # 💾 SAVE IMAGE


                # 🖥️ SHOW FRAME
                cv2.imshow("Detection", frame)
                cv2.waitKey(1000)



                time.sleep(2)

            else:
                print("No objects detected")
                last_spoken = ""
            image_path = "capture.jpg"
            cv2.imwrite(image_path, frame)
                            # 🌐 IMAGGA
            speak("Analyzing image")
            imagga_describe(image_path)
            time.sleep(2)

        else:
            last_spoken = ""

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopped by user")

# -------------------------------------------------
# 🧹 CLEANUP
# -------------------------------------------------
GPIO.cleanup()
cv2.destroyAllWindows()
picam2.stop()
