import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

cascade_path = './haarcascade_frontalface_alt.xml'

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def face_detect(image, image_tmp):
    # image padding
    padding_size = int(image.shape[1] / 2)
    padding_img = cv2.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = cv2.copyMakeBorder(image_tmp, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = image_tmp.astype('float64')

    # face detect
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # face overlay
    if len(facerect) > 0:
        for rect in facerect:
            face_size = rect[2] * 2
            face_pos_adjust = int(rect[2] * 0.5)
            face_img = cv2.imread('./karaage_icon.png', cv2.IMREAD_UNCHANGED)
            face_img = cv2.resize(face_img, (face_size, face_size))
            mask = face_img[:,:,3]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask / 255.0
            face_img = face_img[:,:,:3]

            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] *= 1 - mask
            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] += face_img * mask

    image_tmp = image_tmp[padding_size:padding_size+image.shape[0], padding_size:padding_size+image.shape[1]]
    image_tmp = image_tmp.astype('uint8')

    return image_tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    parser.add_argument('--mode', type=str, default="pose",
                        help='pose / anime')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image_tmp = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, mode=args.mode)

        if args.mode == 'anime':
            image = face_detect(image, image_tmp)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
