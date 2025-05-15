


### AIM:
To interface a USB/CSI camera module with an edge computing platform (e.g., Raspberry Pi, Jetson Nano, etc.) and implement an occupancy detection system using the Histogram of Oriented Gradients (HOG) algorithm.

### Apparatus/Software Required:
S. No.	Equipment / Software	Specification
1.	Edge Computing Device	Raspberry Pi 4 / Jetson Nano
2.	Camera Module	USB Webcam / Pi Camera Module
3.	Operating System	Raspbian OS / Ubuntu
4.	Programming Language	Python 3.x
5.	Libraries	OpenCV, imutils, NumPy
6.	Display Output	HDMI Monitor / VNC Viewer

### Theory:
Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. It counts occurrences of gradient orientation in localized portions of an image. HOG descriptors are particularly useful for detecting humans (pedestrians) in static images or video frames.

Steps involved in HOG-based Occupancy Detection:

Capture frames from the camera.

Resize and preprocess the image.

Use a pre-trained HOG descriptor with a linear SVM to detect people in the image.

Annotate the image with bounding boxes where people are detected.

Display or store the result.

Circuit Diagram / Setup:
Connect the USB camera to the edge computer via a USB port.

Power on the edge device and boot into the OS.

Ensure necessary Python libraries are installed.

### Procedure:
Set up the edge device with a monitor or SSH/VNC connection.

Connect and verify the camera using commands like ls /dev/video* or vcgencmd get_camera.

Install required libraries:

bash
Copy
Edit
pip install opencv-python imutils numpy
Write the Python code to initialize the camera and implement the HOG algorithm.

Run the code and verify that the system detects human presence and draws bounding boxes.

 ###  Python Code:
~~~
import cv2

# --- Ground Truth Input ---
expected_humans = int(input("Enter the actual number of humans visible to the camera: "))

# Initialize USB camera
cap = cv2.VideoCapture(0)

# Check if camera is accessible
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for performance
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8))
    detected_count = len(boxes)

    # Accuracy Calculation
    if expected_humans > 0:
        accuracy = min(detected_count, expected_humans) / expected_humans * 100
    else:
        accuracy = 100.0 if detected_count == 0 else 0.0

    # Print to Console
    print(f"Humans Detected: {detected_count} | Accuracy: {accuracy:.2f}%")

    # Draw detection boxes and info on frame
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"Detected: {detected_count} | Accuracy: {accuracy:.2f}%"
    cv2.putText(frame_resized, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Human Detection", frame_resized)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
~~~
### SCREEN SHOTS OF OUTPUT 

![Screenshot 2025-05-15 114642](https://github.com/user-attachments/assets/fd5fd2a8-5dae-4e4c-9075-5c1765408176)
### RASPI INTERFACE 

![Screenshot 2025-05-15 115011](https://github.com/user-attachments/assets/13328d5e-ff2d-493e-81f1-e1e65e9c93cb)

### Result:
Occupancy detection using the HOG algorithm was successfully implemented. The system was able to identify and highlight human presence in real-time video streams.
