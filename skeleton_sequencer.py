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
volume = 127

note_list_0 = []
note_list_1 = []
note_list_2 = []

play_mode = 'sequencer'

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

def skeleton_sequencer(src):
    global start_time
    global dot_line
    global note_list_0
    global note_list_1
    global note_list_2

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
        start_time += speed
        dot_line += 1
        if dot_line > w_max-1:
            dot_line = 0

        # sound off
        for note in note_list_0:
            midiOutput.note_off(note, volume, 2)
        for note in note_list_1:
            midiOutput.note_off(note, volume, 3)
        for note in note_list_2:
            midiOutput.note_off(note, volume, 4)

        # sound on
        note_list_0 = []
        note_list_1 = []
        note_list_2 = []

        for y in range(0, h_max):
            human_check_0 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_0
            human_check_1 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_1
            human_check_2 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_2

            if human_check_0:
                note_list_0.append(get_pentatonic_scale(y))
            if human_check_1:
                note_list_1.append(get_pentatonic_scale(y))
            if human_check_2:
                note_list_2.append(get_pentatonic_scale(y))


        for note in note_list_0:
            midiOutput.note_on(note, volume, 2)
        for note in note_list_1:
            midiOutput.note_on(note, volume, 3)
        for note in note_list_2:
            midiOutput.note_on(note, volume, 4)


    # draw dot
    for y in range(0, h_max):
        for x in range(0, w_max):
            center = (int(x * d_circle + d_circle * 0.5), int(y * d_circle + d_circle * 0.5))
            if x == dot_line:
                human_check_0 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_0
                human_check_1 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_1
                human_check_2 = dot_color[dot_line][y].tolist() == TfPoseEstimator.HUMAN_COLOR_2

                if human_check_0 or human_check_1 or human_check_2:
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
    parser.add_argument('--fullscreen', type=bool, default=False)
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
            print('midi id=' + str(i))
            midiOutput = pygame.midi.Output(i)

    midiOutput.set_instrument(1, 2)
    midiOutput.set_instrument(1, 3)
    midiOutput.set_instrument(1, 4)

    window_name = 'Skeleton Sequencer'
    if args.fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    while True:
        ret_val, image = cam.read()
        image = cv2.flip(image, 1)

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, mode=play_mode)
        if play_mode == 'sequencer':
            image = skeleton_sequencer(image)

        if play_mode == 'pose':
            cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow(window_name, image)
        logger.debug('show+')

        fps_time = time.time()
        key = cv2.waitKey(1)
        if key == 27: # ESC key
            break
        if key == ord('s') or key == ord('S'):
            play_mode = 'sequencer'
        if key == ord('m') or key == ord('M'):
            for note in note_list_0:
                midiOutput.note_off(note, volume, 2)
            for note in note_list_1:
                midiOutput.note_off(note, volume, 3)
            for note in note_list_2:
                midiOutput.note_off(note, volume, 4)
            play_mode = 'pose'

        if key == ord('0'):
            midiOutput.set_instrument(1, 2)
            midiOutput.set_instrument(1, 3)
            midiOutput.set_instrument(1, 4)
            TfPoseEstimator.HUMAN_COLOR_0 = [255, 0, 0]
            TfPoseEstimator.HUMAN_COLOR_1 = [0, 255 , 0]
            TfPoseEstimator.HUMAN_COLOR_2 = [0, 0, 255]
        if key == ord('1'):
            midiOutput.set_instrument(13, 2)
            midiOutput.set_instrument(22, 3)
            midiOutput.set_instrument(33, 4)
            TfPoseEstimator.HUMAN_COLOR_0 = [128, 128, 128]
            TfPoseEstimator.HUMAN_COLOR_1 = [128, 128 , 0]
            TfPoseEstimator.HUMAN_COLOR_2 = [0, 128, 0]
        if key == ord('2'):
            midiOutput.set_instrument(49, 2)
            midiOutput.set_instrument(58, 3)
            midiOutput.set_instrument(73, 4)
            TfPoseEstimator.HUMAN_COLOR_0 = [255, 0, 255]
            TfPoseEstimator.HUMAN_COLOR_1 = [0, 128 , 128]
            TfPoseEstimator.HUMAN_COLOR_2 = [128, 0, 128]
        if key == ord('3'):
            midiOutput.set_instrument(49, 2)
            midiOutput.set_instrument(58, 3)
            midiOutput.set_instrument(73, 4)
            TfPoseEstimator.HUMAN_COLOR_0 = [0, 0, 128]
            TfPoseEstimator.HUMAN_COLOR_1 = [255, 255 , 0]
            TfPoseEstimator.HUMAN_COLOR_2 = [0, 0, 0]

        logger.debug('finished+')

    for note in note_list_0:
        midiOutput.note_off(note, volume, 2)
    for note in note_list_1:
        midiOutput.note_off(note, volume, 3)
    for note in note_list_2:
        midiOutput.note_off(note, volume, 4)

    cv2.destroyAllWindows()
    midiOutput.close()
