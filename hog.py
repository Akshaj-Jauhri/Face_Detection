"""
USAGE:
python hog.py -i input/image1.jpg
python hog.py -i input/image1.jpg -u 1
python hog.py -h
"""

import dlib
import argparse
import cv2
import sys
import time

import process_dlib_boxes

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../../input/test_data/image1.jpg',
                    help='path to the input image')
parser.add_argument('-u', '--upsample', default=None, type=float,
                    help='factor by which to upsample the image, default None, ' + \
                          'pass 1, 2, 3, ...')
args = vars(parser.parse_args())

# read the image and convert to RGB color format
image = cv2.imread(args['input'])
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# path for saving the result image
save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_u{args['upsample']}.jpg"

# initilaize the Dlib face detector according to the upsampling value
detector = dlib.get_frontal_face_detector()

# carry out the face detection in the image
if args['upsample'] == None:
    start = time.time()
    detected_boxes = detector(image_rgb)
    end = time.time()
elif args['upsample'] > 0 and args['upsample'] < 4:
    start = time.time()
    detected_boxes = detector(image_rgb, int(args['upsample']))
    end = time.time()
# prematurely exit the program when upsample value is >= 4
else:
    warn_string = 'Please provide usample value > 1 and < 4.' + \
             ' Else it might lock your CPU.'
    print(warn_string)
    sys.exit(0)

# process the detection boxes
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