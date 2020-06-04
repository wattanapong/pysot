import os
import cv2
import re
import numpy as np
import json

from glob import glob
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import get_min_max_bbox, center2corner, Center
from pysot.datasets.augmentation import Augmentation

class Video(object):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False, config=None):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        self.img_names = [os.path.join(root, x.replace('color/','')) for x in img_names]
        self.imgs = None
        self.config = config

        self.template_aug = Augmentation(
            config.DATASET.TEMPLATE.SHIFT,
            config.DATASET.TEMPLATE.SCALE,
            config.DATASET.TEMPLATE.BLUR,
            config.DATASET.TEMPLATE.FLIP,
            config.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            config.DATASET.SEARCH.SHIFT,
            config.DATASET.SEARCH.SCALE,
            config.DATASET.SEARCH.BLUR,
            config.DATASET.SEARCH.FLIP,
            config.DATASET.SEARCH.COLOR
        )

        # create anchor target
        self.anchor_target = AnchorTarget()

        if load_img:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]
        else:
            img = cv2.imread(self.img_names[0])
            assert img is not None, self.img_names[0]
            self.width = img.shape[1]
            self.height = img.shape[0]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print(name, len(pred_traj), len(self.gt_traj), self.name)
                if store:
                    self.pred_trajs[name] = pred_traj
                else:
                    return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    def load_img(self):
        if self.imgs is None:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]

    def free_img(self):
        self.imgs = None

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.config.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            img = cv2.imread(self.img_names[idx])
        else:
            img = self.imgs[idx]

        return img, self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):

            if self.imgs is not None:
                img = self.imgs[i]
            else:
                img = cv2.imread(self.img_names[i])

            # gray = self.config.DATASET.GRAY and self.config.DATASET.GRAY > np.random.random()
            gray = False

            # get Corner
            bbox = get_min_max_bbox(np.asarray(self.gt_traj[i], dtype=np.float32))
            bbox = self._get_bbox(img, bbox[-2:])

            # augmentation
            template, _ = self.template_aug(img,
                                            bbox,
                                            self.config.TRAIN.EXEMPLAR_SIZE,
                                            gray=gray)

            search, bbox_s = self.search_aug(img,
                                           bbox,
                                           self.config.TRAIN.SEARCH_SIZE,
                                           gray=gray)

            # get labels
            cls, delta, delta_weight, overlap = self.anchor_target(
                bbox_s, self.config.TRAIN.OUTPUT_SIZE)
            cls_s, delta_s, delta_weight_s, overlap_s = self.anchor_target(
                bbox_s, self.config.TRAIN.OUTPUT_SIZE)

            yield img, self.gt_traj[i], cls, delta, delta_weight, bbox, cls_s, delta_s, delta_weight_s, bbox_s

    def draw_box(self, roi, img, linewidth, color, name=None):
        """
            roi: rectangle or polygon
            img: numpy array img
            linewith: line width of the bbox
        """
        if len(roi) > 6 and len(roi) % 2 == 0:
            pts = np.array(roi, np.int32).reshape(-1, 1, 2)
            color = tuple(map(int, color))
            img = cv2.polylines(img, [pts], True, color, linewidth)
            pt = (pts[0, 0, 0], pts[0, 0, 1]-5)
            if name:
                img = cv2.putText(img, name, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        elif len(roi) == 4:
            if not np.isnan(roi[0]):
                roi = list(map(int, roi))
                color = tuple(map(int, color))
                img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),
                         color, linewidth)
                if name:
                    img = cv2.putText(img, name, (roi[0], roi[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        return img

    def show(self, pred_trajs={}, linewidth=2, show_name=False):
        """
            pred_trajs: dict of pred_traj, {'tracker_name': list of traj}
                        pred_traj should contain polygon or rectangle(x, y, width, height)
            linewith: line width of the bbox
        """
        assert self.imgs is not None
        video = []
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        colors = {}
        if len(pred_trajs) == 0 and len(self.pred_trajs) > 0:
            pred_trajs = self.pred_trajs
        for i, (roi, img) in enumerate(zip(self.gt_traj,
                self.imgs[self.start_frame:self.end_frame+1])):
            img = img.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = self.draw_box(roi, img, linewidth, (0, 255, 0),
                    'gt' if show_name else None)
            for name, trajs in pred_trajs.items():
                if name not in colors:
                    color = tuple(np.random.randint(0, 256, 3))
                    colors[name] = color
                else:
                    color = colors[name]
                img = self.draw_box(trajs[0][i], img, linewidth, color,
                        name if show_name else None)
            cv2.putText(img, str(i+self.start_frame), (5, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
            cv2.imshow(self.name, img)
            cv2.waitKey(40)
            video.append(img.copy())
        return video
