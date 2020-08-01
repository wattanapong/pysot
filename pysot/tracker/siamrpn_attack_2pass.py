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

torch.manual_seed(1999)
np.random.seed(1999)

class SiamRPNAttack2Pass(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNAttack2Pass, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.zf = None
        self.zfa = None

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

        anchor = torch.from_numpy(anchor).cuda()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]

        delta[2, :] = torch.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * anchor[:, 3]

        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score_softmax = F.softmax(score, dim=1)[:, 1]
        return score_softmax

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

    # min overlap
    def l1_loss(self, pred_box_a, pred_box, lr):
        a = 0.3
        b = 0.7

        xa, ya, wa, ha = pred_box_a
        x, y, w, h = pred_box
        wa = torch.tensor(self.size[0]).cuda() * (1 - lr) + wa * lr
        ha = torch.tensor(self.size[1]).cuda() * (1 - lr) + ha * lr
        c_loss = -1 * torch.sum(torch.sqrt(xa**2+ya**2))
        # shape_loss = -1 * torch.sum(torch.sqrt((wa - w)**2+(ha - h)**2))
        # return c_loss + shape_loss
        return c_loss

    # min confident score
    def l2_loss(self, score, sort_idx, th):
        confidence_loss = torch.sum(score[sort_idx[:th]]) - torch.sum(score[sort_idx[th * 2:th * 3]])
        return confidence_loss

    def l3_loss(self, z_crop, z_crop_a):
        z_energy = torch.norm(z_crop.cuda() - z_crop_a)
        return z_energy


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


    def init(self, img, bbox, attacker=None, epsilon=0, update=True):

        if update or attacker is None:
            self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
            self.size = np.array([bbox[2], bbox[3]])

            # calculate z crop size
            w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z = round(np.sqrt(w_z * h_z))

            # self.channel_average = torch.mean(img, dim=(0, 1))
            self.channel_average = np.mean(img, axis=(0, 1))

            self.z_crop, box, pad = self.get_subwindow_custom(img, self.center_pos,
                                            cfg.TRACK.EXEMPLAR_SIZE,
                                            s_z, self.channel_average)

            h, w, _ = img.shape

            box[0] = box[0] - pad[0]
            box[1] = box[1] - pad[0]
            box[2] = box[2] - pad[2]
            box[3] = box[3] - pad[2]

            box[0] = 0 if box[0] < 0 else box[0]
            box[2] = 0 if box[2] < 0 else box[2]
            box[1] = h - 1 if box[1] > h else box[1]
            box[3] = w - 1 if box[3] > w else box[3]

        if attacker is None:
            self.model.template(self.z_crop, epsilon=0)
            self.zf = torch.mean(torch.stack(self.model.zf), 0)
        else:
            self.z_crop_adv = attacker.template(self.z_crop, self.model, epsilon)
            self.zf = torch.mean(torch.stack(attacker.zf), 0)

        if update:
            return s_z, box, pad

    def train(self, img, attacker=None, bbox=None, epsilon=0, idx=0, batch=200, debug=False):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        self.x_crop, _, _ = self.get_subwindow_custom(img, (bbox[0], bbox[1]), cfg.TRACK.INSTANCE_SIZE, round(s_x),
                                           self.channel_average, shift=32)
        # self.z_crop_adv = attacker.template(self.z_crop, self.model, epsilon)
        # self.zfa = torch.mean(torch.stack(attacker.zf), 0)

        outputs = attacker(self.x_crop, self.model, iter)

        score_softmax = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # max_score.values and max_score.indices
        _, sort_idx = torch.sort(-score_softmax)

        def change(r):
            return torch.max(r, 1./r)

        def sz(w, h):
            if not torch.is_tensor(w):
                w = torch.tensor(w).type(torch.float)
            if not torch.is_tensor(h):
                h = torch.tensor(h).type(torch.float)
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change(torch.tensor(self.size[0]/self.size[1]).type(torch.float) / (pred_bbox[2, :]/pred_bbox[3, :]))

        penalty = torch.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K).cuda()

        lr = (penalty[sort_idx[0]] * score_softmax[sort_idx[0]] * cfg.TRACK.LR).data

        l1 = self.l1_loss(pred_bbox[:, sort_idx[0]], bbox, lr)

        l2 = self.l2_loss(score_softmax, sort_idx, 45)
        if attacker.template_average is None:
            l3 = None
        else:
            l3 = self.l3_loss(attacker.template_average, attacker.adv_z)

        return {
            'l1': l1,
            'l2': l2,
            'l3': l3
        }

    def track(self, img, attacker=None, epsilon=0, idx=0, iter=0, debug=False):

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop, _, _ = self.get_subwindow_custom(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop, iter)

        score_softmax = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return torch.max(r, 1./r)

        def sz(w, h):
            if not torch.is_tensor(w):
                w = torch.tensor(w).type(torch.float)
            if not torch.is_tensor(h):
                h = torch.tensor(h).type(torch.float)
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change(torch.tensor(self.size[0]/self.size[1]).type(torch.float) / (pred_bbox[2, :]/pred_bbox[3, :]))

        penalty = torch.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K).cuda()
        pscore_softmax = penalty * score_softmax

        # window penalty
        pscore_softmax = pscore_softmax * torch.tensor(1 - cfg.TRACK.WINDOW_INFLUENCE, dtype=torch.float32).cuda() + \
        torch.tensor(self.window * cfg.TRACK.WINDOW_INFLUENCE, dtype=torch.float32).cuda()

        _, sort_idx = torch.sort(-pscore_softmax)
        # if attacker is not None:
        #     pdb.set_trace()

        best_idx = sort_idx[0]
        bbox = pred_bbox[:, best_idx].data.cpu().numpy() / scale_z

        best_score = score_softmax[best_idx]
        lr = (penalty[best_idx] * best_score * cfg.TRACK.LR).data.cpu().numpy()

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
            'best_score': score_softmax[sort_idx[0]],
            'target_score': score_softmax[sort_idx[45 - 1]],
            'center_pos': np.array([cx, cy]),
            'size': np.array([width, height])
        }

    def get_subwindow_custom(self, im, pos, model_sz, original_sz, avg_chans, shift=0):
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
        offset = (np.random.rand(1) * 2*shift - shift)
        pos = pos + offset
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
            if torch.is_tensor(im):
                te_im = torch.zeros(size, dtype=torch.uint8)
            else:
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

        if torch.is_tensor(im_patch):
            im_patch = im_patch.data.cpu().numpy()

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