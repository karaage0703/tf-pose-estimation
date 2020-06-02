import logging
import math

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.tensblur.smoother import Smoother
import tensorflow.contrib.tensorrt as trt

try:
    from tf_pose.pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.handlers.clear()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


def face_overlay(image, image_tmp, cascade):
    # image padding
    padding_size = int(image.shape[1] / 2)
    padding_img = cv2.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = cv2.copyMakeBorder(image_tmp, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = image_tmp.astype('float64')

    # face detect
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None, trt_bool=False):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if trt_bool is True:
            output_nodes = ["Openpose/concat_stage7"]
            graph_def = trt.create_inference_graph(
                graph_def,
                output_nodes,
                max_batch_size=1,
                max_workspace_size_bytes=1 << 20,
                precision_mode="FP16",
                # precision_mode="INT8",
                minimum_segment_size=3,
                is_dynamic_op=True,
                maximum_cached_engines=int(1e3),
                # use_calibration=True,
            )

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print(ts)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        if trt_bool is True:
            smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0, 19)
        else:
            smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )

        # logs
        if self.tensor_image.dtype == tf.quint8:
            logger.info('quantization mode enabled.')

    def __del__(self):
        # self.persistent_sess.close()
        pass

    def get_flops(self):
        flops = tf.profiler.profile(self.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        return flops.total_float_ops

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False, mode='pose'):
        human_color = [255, 255, 255]

        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]

        # black background
        if mode == 'anime':
            npimg = np.zeros((image_h, image_w, 3), np.uint8)

        centers = {}
        for human in humans:
            rshoulder_pos = None
            lshoulder_pos = None
            rhip_pos = None
            lhip_pos = None

            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                # cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

                if mode == 'pose':
                    cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
                if mode == 'anime':
                    if i == 4 or i == 7: # draw wrist
                        cv2.circle(npimg, center, 50, human_color, thickness=-1, lineType=8, shift=0)
                    if i == 10 or i == 13: # draw ankle
                        cv2.circle(npimg, center, 60, human_color, thickness=-1, lineType=8, shift=0)
                    if i == 2: # Store RShoulder position
                        rshoulder_pos = center
                    if i == 5: # Store LShoulder position
                        lshoulder_pos = center
                    if i == 8: # Store RHip position
                        rhip_pos = center
                    if i == 11: # Store LHip position
                        lhip_pos = center

            # draw body
            if rshoulder_pos != None and lshoulder_pos != None and rhip_pos != None and lhip_pos != None:
                body_center = ((rshoulder_pos[0] + lshoulder_pos[0] + rhip_pos[0] + lhip_pos[0]) / 4,
                               (rshoulder_pos[1] + lshoulder_pos[1] + rhip_pos[1] + lhip_pos[1])/ 4)

                body_size = (((lhip_pos[0] - rhip_pos[0]) / 2 + (lshoulder_pos[0] - rshoulder_pos[0]) / 2) * 1.5,
                             ((rhip_pos[1] + lhip_pos[1]) / 2 - (rshoulder_pos[1] + lshoulder_pos[1]) / 2) * 1.5)

                cv2.ellipse(npimg, ((body_center), (body_size), 0), human_color, thickness=-1, lineType=8)

            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                if mode == 'pose':
                    # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                    cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                if mode == 'anime':
                    if pair_order < 13: # under neck
                        cv2.line(npimg, centers[pair[0]], centers[pair[1]], human_color, 50)

        return npimg

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_image: [img], self.upsample_size: upsample_size
            })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


if __name__ == '__main__':
    import pickle

    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t;
    t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
