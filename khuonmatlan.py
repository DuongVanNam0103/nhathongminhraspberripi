from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np

# Setup GPIO for servo motor
SERVO_PIN = 17  # Pin to control the servo motor
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set up PWM for the servo motor
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
pwm.start(0)

# Function to set servo angle
def set_servo_angle(angle):
    duty = angle / 18 + 2  # Convert angle to duty cycle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
encodingsP = "encodings.pickle"
cascade = "haarcascade_frontalface_default.xml"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

prevTime = 0
doorUnlock = False
consecutive_correct_detections = 0
threshold = 3  # Số lần nhận diện đúng liên tiếp trước khi mở cửa

# Loop over frames from the video file stream
while True:
    # Grab the frame from the video stream and resize it to 500px
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Convert the input frame from BGR to grayscale (for face detection) and from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Reorder bounding box coordinates
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to known encodings with reduced tolerance
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
        name = "Unknown"  # Default if face is not recognized

        if True in matches:
            # Find the face distance (confidence score)
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = np.argmin(face_distances)

            # If the best match has a low enough distance, consider it a valid match
            if face_distances[best_match_index] < 0.4:
                name = data["names"][best_match_index]
                
                # Increase the correct detection count if we recognize the face correctly
                consecutive_correct_detections += 1
                if consecutive_correct_detections >= threshold:
                    # Rotate the servo to 90 degrees (unlock)
                    set_servo_angle(90)
                    prevTime = time.time()
                    doorUnlock = True
                    print(f"Servo turned to 90 degrees for {name}")
                    consecutive_correct_detections = 0  # Reset the counter
            else:
                consecutive_correct_detections = 0
        else:
            consecutive_correct_detections = 0

        # Update the list of names
        names.append(name)

    # Lock the servo after 5 seconds by rotating back to 0 degrees
    if doorUnlock == True and time.time() - prevTime > 5:
        doorUnlock = False
        set_servo_angle(0)  # Rotate servo back to 0 degrees
        print("Servo turned back to 0 degrees")

    # Display the recognized face name on the screen
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when 'q' key is pressed
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
pwm.stop()
GPIO.cleanup()
