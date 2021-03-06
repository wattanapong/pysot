# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import torch, pdb, os

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import get_min_max_bbox, center2corner, Center, get_axis_aligned_bbox

class SiamRPNAttackTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNAttackTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

        self.anchor_target = AnchorTarget()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[-2:]
        cx, cy = imw // 2, imh // 2

        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape

        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z

        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def perturb(self, bbox, sz):
        # bbox[1] = bbox[1] - bbox[3] + sz/5
        # bbox[2] = bbox[2] // 6
        # bbox[3] = bbox[3] // 6
        return bbox

    def crop(self, img, bbox=None, im_name=None):
        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # [x, y, w, h] to [cx, cy, w, h]
        bbox = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]]
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        sz = round(s_x)
        # s_x = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        limit_size = cfg.TRACK.INSTANCE_SIZE

        # get crop
        # box = {top, bottom, left, right}
        # pad = {pad top, pad bottom, pad left, pad pad right}
        h, w, _ = img.shape
        _crop, box, pad = self.get_subwindow_custom(img, self.center_pos, limit_size, sz, self.channel_average)

        box[0] = box[0] - pad[0]
        box[1] = box[1] - pad[0]
        box[2] = box[2] - pad[2]
        box[3] = box[3] - pad[2]

        box[0] = 0 if box[0] < 0 else box[0]
        box[2] = 0 if box[2] < 0 else box[2]
        box[1] = h - 1 if box[1] > h else box[1]
        box[3] = w - 1 if box[3] > w else box[3]

        return _crop, sz, box, pad

    # def crop(self, img, bbox=None, im_name=None):
    #     # calculate channel average
    #     self.channel_average = np.mean(img, axis=(0, 1))
    #
    #     # [x, y, w, h] to [cx, cy, w, h]
    #     bbox = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]]
    #     w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     s_z = round(np.sqrt(w_z * h_z))
    #     s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    #     sz = round(s_x)
    #     # s_x = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    #     limit_size = cfg.TRACK.INSTANCE_SIZE
    #
    #     # get crop
    #     # box = {top, bottom, left, right}
    #     # pad = {pad top, pad bottom, pad left, pad pad right}
    #     _crop, box, pad = self.get_subwindow_custom(img, self.center_pos, limit_size, sz, self.channel_average)
    #
    #     h = box[1] - box[0]
    #     w = box[3] - box[2]
    #     _bbox = self._get_bbox(_crop, [w, h])
    #
    #     bbox_perturb = self.perturb(_bbox, cfg.TRAIN.SEARCH_SIZE)
    #     cls, delta, delta_w, overlap = self.anchor_target(
    #         bbox_perturb, cfg.TRAIN.OUTPUT_SIZE)
    #     bp = np.array(bbox_perturb, np.int)
    #
    #     _img = _crop.data.cpu().numpy().squeeze().transpose([1, 2, 0])
    #     _img = cv2.UMat(_img).get()
    #
    #     cv2.rectangle(_img, (bp[0], bp[1]), (bp[2], bp[3]), (0, 0, 0), 3)
    #     cv2.imwrite('/media/wattanapongsu/3T/temp/save/bag/crop' + im_name + '.jpg', _img)
    #
    #     pdb.set_trace()
    #
    #     box[0] = box[0] - pad[0]
    #     box[1] = box[1] - pad[0]
    #     box[2] = box[2] - pad[2]
    #     box[3] = box[3] - pad[2]
    #
    #     box[0] = 0 if box[0] < 0 else box[0]
    #     box[2] = 0 if box[2] < 0 else box[2]
    #     box[1] = h - 1 if box[1] > h else box[1]
    #     box[3] = w - 1 if box[3] > w else box[3]
    #
    #     # if im_name is not None and True:
    #     #     _img = _crop.data.cpu().numpy().squeeze().transpose([1, 2, 0])
    #     #     _img = cv2.UMat(_img).get()
    #     #
    #     #     cv2.rectangle(_img, (bp[0], bp[1]), (bp[2], bp[3]), (0, 0, 0), 3)
    #     #     cv2.imwrite('/media/wattanapongsu/3T/temp/save/bag/crop' + im_name + '.jpg', _img)
    #
    #     return _crop, sz, box, pad, cls, delta, delta_w

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # get labels
        # bb = np.array([bbox[0],bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]])
        # cls, delta, delta_weight, overlap = self.anchor_target(
        #     bb, cfg.TRAIN.OUTPUT_SIZE)

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        self.z_crop, _, _ = self.get_subwindow_custom(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        # img2 = self.z_crop.data.cpu().numpy().squeeze().transpose([1, 2, 0])
        # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save', 'bag', 'z.jpg'), img2)

        self.model.template(self.z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # img2 = x_crop.data.cpu().numpy().squeeze().transpose([1, 2, 0])
        # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save', 'bag', 'x.jpg'), img2)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

    def get_subwindow_custom(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch, [int(context_ymin), int(context_ymax), int(context_xmin), int(context_xmax)], \
            [top_pad, bottom_pad, left_pad, right_pad]
