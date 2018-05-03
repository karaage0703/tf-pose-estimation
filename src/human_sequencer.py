import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import pygame
import pygame.midi

pygame.init()
pygame.midi.init()

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# time param
start_time = 0
speed = 0.5

# dot param
d_circle = 30
dot_line = 0

# midi setting
instrument = 0
port = 1
volume = 127
 
note_list = []

def get_pentatonic_scale(note):
    # C
    if note%5 == 0:
      out_note = note//5*12

    # D#
    if note%5 == 1:
      out_note = note//5*12 + 3

    # F
    if note%5 == 2:
      out_note = note//5*12 + 5

    # G
    if note%5 == 3:
      out_note = note//5*12 + 7

    # A#
    if note%5 == 4:
      out_note = note//5*12 + 10

    out_note += 60;
    while out_note > 127:
      out_note -= 128

    return out_note

def human_sequencer(src):
    global start_time
    global dot_line
    global note_list

    image_h, image_w = src.shape[:2]

    h_max = int(image_h / d_circle)
    w_max = int(image_w / d_circle)

    # create blank image
    npimg_target = np.zeros((image_h, image_w, 3), np.uint8)
    dot_color = [[0 for i in range(h_max)] for j in range(w_max)] 

    # make dot information from ndarray
    for y in range(0, h_max):
        for x in range(0, w_max):
            dot_color[x][y] = src[y*d_circle][x*d_circle]
 
    # move dot
    current_time = time.time() - start_time
    while time.time() - start_time > speed:
        start_time += speed
        dot_line += 1
        if dot_line > w_max-1:
            dot_line = 0
        
        # sound off
        for note in note_list:
            midiOutput.note_off(note,volume)

        # sound on
        note_list = []
        for y in range(0, h_max):
            if dot_color[dot_line][y][0] == 255:
                note_list.append(get_pentatonic_scale(y))

        for note in note_list:
            midiOutput.note_on(note,volume)


    # draw dot
    for y in range(0, h_max):
        for x in range(0, w_max):
            center = (int(x * d_circle + d_circle * 0.5), int(y * d_circle + d_circle * 0.5))
            if x == dot_line:
                if dot_color[x][y][0] == 255:
                    cv2.circle(npimg_target, center, int(d_circle/2) , [255-(int)(dot_color[x][y][0]),255-(int)(dot_color[x][y][1]),255-(int)(dot_color[x][y][2])] , thickness=-1, lineType=8, shift=0)
                else:
                    cv2.circle(npimg_target, center, int(d_circle/2) , [255,255,255] , thickness=-1, lineType=8, shift=0)
            else:
                cv2.circle(npimg_target, center, int(d_circle/2) , [(int)(dot_color[x][y][0]),(int)(dot_color[x][y][1]),(int)(dot_color[x][y][2])] , thickness=-1, lineType=8, shift=0)

    return npimg_target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    print("midi devices")
    for id in range(pygame.midi.get_count()):
        print(pygame.midi.get_device_info(id))

    midiOutput = pygame.midi.Output(port, 1)
    midiOutput.set_instrument(instrument)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    start_time = time.time()

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image = human_sequencer(image)
    
        logger.debug('show+')
        cv2.imshow('tf-pose-estimation result', image)

        if cv2.waitKey(1) == 27:  # ESC key
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
    del midiOutput
    pygame.midi.quit()