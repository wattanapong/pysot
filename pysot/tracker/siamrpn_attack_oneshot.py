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


class SiamRPNAttackOneShot(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNAttackOneShot, self).__init__()
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

    def _convert_bbox(self, delta, anchor, need_tensor=False):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)

        if not need_tensor:
            delta = delta.data.cpu().numpy()
        else:
            anchor = torch.from_numpy(anchor).cuda()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]

        if not need_tensor:
            delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
            delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        else:
            delta[2, :] = torch.exp(delta[2, :]) * anchor[:, 2]
            delta[3, :] = torch.exp(delta[3, :]) * anchor[:, 3]

        return delta

    def _convert_score(self, score, need_tensor=False):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)

        if not need_tensor:
            score = F.softmax(score, dim=1).data[:, 1]
            score = score.cpu().numpy()
            return score
        else:
            score_softmax = F.softmax(score, dim=1).data[:, 1]
            score = score[:,1]
            return score, score_softmax

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

    def cls_loss(self, pscore, pred_box, sort_idx, scale_z, shape, lr):
        # pred_box = torch.from_numpy(pred_box).cuda()
        # p = [45, 90, 135]
        # print('score update ', target_score, pscore[sort_idx[0]])
        a = 0.5
        b = 1.5
        c = 0.2 # 0.03 #
        a_ = 2
        b_ = -1
        c_ = 20 # 0.1 #

        imh, imw, _ = shape

        top_pred_box = pred_box[:, sort_idx[0]]
        # pdb.set_trace()
        w_inverse = a + b * torch.tanh(c * (pred_box[:2, sort_idx[0:45]] - top_pred_box[:2, None]))
        l1 = torch.sum(pscore[sort_idx[:45]] / w_inverse) - torch.sum(pscore[sort_idx[90:135]])

        w_inverse = a_ + b_ * torch.tanh(c_ * (self.zf_min - self.zf_mean))

        # maximize difference features from adversarial and original image
        l2 = -torch.norm((self.zfa - self.zf)/w_inverse)

        # maximize union of all prediction boxes ( 1 - 45 )

        # pdb.set_trace()
        # idx = 0
        # distance = torch.zeros(45)
        # for i in sort_idx[0:45]:
        #     cx = pred_box[1, i] + self.center_pos[0]
        #     cy = pred_box[0, i] + self.center_pos[1]
        #
        #     # smooth bbox
        #     lr = torch.tensor(lr).type(torch.float)
        #     width = self.size[0] * (1 - lr) + pred_box[3, i] * lr
        #     height = self.size[1] * (1 - lr) + pred_box[2, i] * lr
        #
        #     if idx == 0:
        #         p1 = cx - width / 2, cy - height / 2
        #         p2 = cx + width / 2, cy + height / 2
        #     else:
        #         distance[idx] =
        #
        #
        l3 = -torch.norm(pred_box[:2, sort_idx[0:45]] - top_pred_box[:2, None])
        # pdb.set_trace()

        # w_inverse = a_ + b_ * torch.tanh(c_ * (self.z_crop_min - self.z_crop_mean))
        # minimize pixel variation of adversarial image
        # l3 = torch.sum((self.z_crop - self.z_crop_adv)/w_inverse)

        # minimize width and height prediction box
        # l3 = torch.norm(pred_box[2:4, sort_idx[0:45]])

        # pdb.set_trace()
        return l1, l2, l3

    def cls_loss_oneshot(self, pscore, pred_box, sort_idx, scale_z):
        pred_box = torch.from_numpy(pred_box).cuda()
        # p = [45, 90, 135]
        # print('score update ', target_score, pscore[sort_idx[0]])
        a = 0.5
        b = 1.5
        c = 0.2
        a_ = 2
        b_ = -1
        c_ = 20

        top_pred_box = pred_box[:, sort_idx[0]]
        # pdb.set_trace()
        w_inverse = a + b * torch.tanh(c * torch.sqrt((pred_box[:, sort_idx[0:45]] - top_pred_box[:, None])**2)/scale_z)
        l1 = torch.sum(pscore[sort_idx[:45]] / w_inverse) - torch.sum(pscore[sort_idx[90:135]])

        w_inverse = a_ + b_ * torch.tanh(c_ * (self.zf_min - self.zf_mean))
        l2 = -torch.norm((self.zfa - self.zf)/w_inverse)
        # pdb.set_trace()
        return l1, l2

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

    def init_default(self, img, bbox, attacker=None, epsilon=0):

        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
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
            attacker.template(self.z_crop, self.model, epsilon=0)
            self.zf = torch.mean(torch.stack(attacker.zf), 0)

        # self.model.template(self.z_crop, epsilon=0)
        # self.zf = torch.mean(torch.stack(self.model.zf), 0)

        return s_z, box, pad

    def init(self, img, bbox, attacker=None, epsilon=0):

        # img = torch.from_numpy(img).type(torch.FloatTensor)

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
            attacker.template(self.z_crop, self.model, epsilon=0)
            self.zf = torch.mean(torch.stack(attacker.zf), 0)

        # self.model.template(self.z_crop, epsilon=0)
        # self.zf = torch.mean(torch.stack(self.model.zf), 0)

        return s_z, box, pad

    def track_default(self, img, attacker=None, epsilon=0, zf=None, idx=0, iter=0, debug=False):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop, _, _ = self.get_subwindow_custom(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        if attacker is None:
            outputs = self.model.track(x_crop, iter)
        else:
            self.z_crop_adv = attacker.template(self.z_crop, self.model, epsilon)
            self.zfa = torch.mean(torch.stack(attacker.zf), 0)
            outputs = attacker.track(x_crop, self.model, iter)

        score = self._convert_score(outputs['cls'], False)
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors, False)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        sort_idx = np.argsort(-pscore)

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

        if zf is not None:
            return {
                    'bbox': bbox,
                    'best_score': score[sort_idx[0]],
                    'l1': 0,
                    'l2': 0,
                    'l3': 0,
                    'center_pos': np.array([cx, cy]),
                    'size': np.array([width, height])
                   }
        else:
            return {
                'bbox': bbox,
                'best_score': score[sort_idx[0]],
                'target_score': score[sort_idx[45 - 1]],
                'center_pos': np.array([cx, cy]),
                'size': np.array([width, height])
            }

    def track(self, img, attacker=None, epsilon=0, zf=None, idx=0, iter=0, debug=False):

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop, _, _ = self.get_subwindow_custom(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        if attacker is None:
            outputs = self.model.track(x_crop, iter)
        else:
            self.z_crop_adv = attacker.template(self.z_crop, self.model, epsilon)
            self.zfa = torch.mean(torch.stack(attacker.zf), 0)
            outputs = attacker.track(x_crop, self.model, iter)

        score, score_softmax = self._convert_score(outputs['cls'], True)
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors, True)

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

        best_idx = sort_idx[0]
        bbox = pred_bbox[:, best_idx].data.cpu().numpy() / scale_z

        best_score = score_softmax[best_idx]
        lr = (penalty[best_idx] * best_score * cfg.TRACK.LR).data.cpu().numpy()

        if zf is not None:

            if iter == 0:
                self.zf = zf
                self.zf_min, idx_min = torch.min(zf, 1)
                self.zf_mean = torch.mean(zf, 1)
                self.z_crop_min, _ = torch.min(self.z_crop, 1)
                self.z_crop_mean = torch.mean(self.z_crop, 1)

            if debug:
                img2 = self.z_crop_adv.data.cpu().numpy().squeeze().transpose([1, 2, 0])
                cv2.imwrite(os.path.join('/media/wattanapongsu/4T/temp/save', 'bag',
                                     'z'+str(idx).zfill(6)+'_'+str(iter).zfill(2)+'.jpg'), img2)

            l1, l2, l3 = self.cls_loss(pscore_softmax, pred_bbox, sort_idx, scale_z, img.shape, lr)
        else:
            if debug:
                img2 = self.z_crop.data.cpu().numpy().squeeze().transpose([1, 2, 0])
                cv2.imwrite(os.path.join('/media/wattanapongsu/4T/temp/save', 'bag',
                                     'z' + str(idx).zfill(6) + '.jpg'), img2)

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # update state
        if attacker is None:
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if zf is not None:
            return {
                    'bbox': bbox,
                    'best_score': score[sort_idx[0]],
                    'l1': l1,
                    'l2': l2,
                    'l3': l3,
                    'center_pos': np.array([cx, cy]),
                    'size': np.array([width, height])
                   }
        else:
            return {
                'bbox': bbox,
                'best_score': score[sort_idx[0]],
                'target_score': score[sort_idx[45 - 1]],
                'center_pos': np.array([cx, cy]),
                'size': np.array([width, height])
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