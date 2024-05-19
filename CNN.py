"""
USAGE:
python face_det_image.py -i input/image_1.jpg
python face_det_image.py -i input/image_1.jpg -u 1
python face_det_image.py -h
"""

import dlib
import argparse
import cv2
import time

import process_dlib_boxes

# contruct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg',
                    help='path to the input image')
parser.add_argument('-u', '--upsample', default=None, type=int,
                    help='factor by which to upsample the image, default None, ' + \
                          'pass 1, 2, 3, ...')
args = vars(parser.parse_args())

# read the image and convert to RGB color format
image = cv2.imread(args['input'])
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# path for saving the result image
save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_u{args['upsample']}.jpg"

# initilaize the Dlib face detector
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

if args['upsample'] == None:
    start = time.time()
    detected_boxes = detector(image)
    end = time.time()
elif args['upsample'] > 0:
    start = time.time()
    detected_boxes = detector(image, int(args['upsample']))
    end = time.time()

# process the detection boxes and draw them around faces
for box in detected_boxes:
    res_box = process_dlib_boxes.process_boxes(box)
    cv2.rectangle(image, (res_box[0], res_box[1]),
                  (res_box[2], res_box[3]), (0, 255, 0), 
                  2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.imwrite(save_name, image)
print(f"Total faces detected: {len(detected_boxes)}")
print(f"Total time taken: {end-start:.3f} seconds.")
print(f"FPS: {1/(end-start):.3f}")