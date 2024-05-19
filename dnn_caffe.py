import cv2
import time

# Define the path to the model files and set the confidence threshold
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt.txt"
conf_threshold = 0.5

# Load the pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Read the input image
frameOpencvDnn = cv2.imread("testimg.jpg")

# Resize the image to match the input size of the model
frameWidth = frameOpencvDnn.shape[1]
frameHeight = frameOpencvDnn.shape[0]
blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

# Set the input to the network and perform inference
net.setInput(blob)

# Measure the time taken for inference
start_time = time.time()
detections = net.forward()
end_time = time.time()
elapsed_time = end_time - start_time

# Iterate over the detections and extract bounding boxes
bboxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        bboxes.append((x1, y1, x2, y2))

# Draw bounding boxes on the image
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Detection", frameOpencvDnn)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Output the number of faces detected and the time taken
print(f"Number of faces detected: {len(bboxes)}")
print(f"Time taken for inference: {elapsed_time:.2f} seconds")
