import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator_hs import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

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

fps_time = 0

# time param
start_time = 0.0
speed = 0.5

# dot param
d_circle = 30
dot_line = 0

# midi setting
instrument = 1 # 0 vocaloid / 1 piano / 9 drum
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
    while time.time() - start_time > speed:
        print(time.time() -start_time)
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
            midiOutput.note_on(note,volume,instrument)


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

    parser.add_argument('--resize', type=str, default='320x176',
                        help='if provided, resize images before they are processed. default=320x176, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('-d', '--device', default='normal_cam') # normal_cam /jetson_nano_raspi_cam
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')

    if args.device == 'normal_cam':
        cam = cv2.VideoCapture(0)
    elif args.device == 'jetson_nano_raspi_cam':
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 \
            ! nvvidconv ! video/x-raw, width=(int)1280, height=(int)960, format=(string)BGRx \
            ! videoconvert \
            ! appsink'
        cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # midi setup
    for i in range(pygame.midi.get_count()):
        interf, name, input_dev, output_dev, opened = pygame.midi.get_device_info(i)
        if output_dev and b'NSX-39 ' in name:
            print(i)
            midiOutput = pygame.midi.Output(i)

    # midiOutput.set_instrument(instrument)

    start_time = time.time()
    while True:
        ret_val, image = cam.read()
        image = cv2.flip(image, 1)

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image = human_sequencer(image)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('Human Sequencer', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27: # ESC key
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
    midiOutput.close()
