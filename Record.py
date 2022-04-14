import cv2
import argparse
import os
from datetime import datetime

# ---------- How to use: ----------
# Run record.py --output [folder]
# Press s to take a snapshot
# Press v to turn of/off continuous recording

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
                default="Imgs",
                help="Output directory")
args = vars(ap.parse_args())

directories = [args["output"], args["output"] + '/data',
               args["output"] + '/data/col', args["output"] + '/data/masks']
for d in directories:
    if not os.path.exists(d):
        os.makedirs(d)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print('Could not open camera')
else:
    running = True
    video = False
    blink = 0
    while running:
        retval = cam.grab()
        if not retval:
            break
        timestamp = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S_%f")
        retval, frame = cam.retrieve()
        display_frame = frame.copy()
        frame = cv2.resize(frame, (640, 480))
        if not retval:
            break
        if video:
            if blink < 15:
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), cv2.FILLED)
            cv2.imwrite(args["output"] + f'/data/col/frame_{timestamp}.png', frame)
        cv2.imshow('Display', display_frame)
        key = cv2.waitKey(1)
        blink = blink+1 if blink < 30 else 0
        if key == ord('q'):
            break
        if key == ord('s') and not video: # snapshot
            cv2.imwrite(args["output"] + f'/data/col/frame_{timestamp}.png', frame)
        if key == ord('v'): # video
            video = not video