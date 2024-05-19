import cv2
import sys
import time

# Reading the image
img = cv2.imread('testimg.jpg')

# Define the scaling factor (0.5 for 50% scale down)
scale_factor = 1

# Resize the image
img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

# Converting resized image to grayscale
gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')

# Start time for face detection
start_time = time.time()

# Applying the face detection method on the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

# End time for face detection
end_time = time.time()

# Count the number of faces detected
num_faces = len(faces_rect)

# Iterating through rectangles of detected faces
for (x, y, w, h) in faces_rect:
    # Scaling the coordinates back to the original image size
    x_orig = int(x / scale_factor)
    y_orig = int(y / scale_factor)
    w_orig = int(w / scale_factor)
    h_orig = int(h / scale_factor)
    cv2.rectangle(img, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Detected faces', img)
cv2.waitKey(0)

# Output the number of faces detected and the time taken for face detection
print(f"Total faces detected: {num_faces}")
print(f"Time taken for face detection: {end_time - start_time:.4f} seconds")
