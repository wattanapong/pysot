# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import logging

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import pdb
from torchvision.transforms import ToTensor, ToPILImage

from torch.utils.data import DataLoader
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from pysot.models.steath import Steath, WeightClipper
import torch.optim as optim

logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--dataset_dir', type=str,
        default='/media/wattanapongsu/3T/dataset',
        help='dataset directory')
parser.add_argument('--savedir', default='', type=str,
        help='save images and videos in this directory')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--epsilon', default='0.1', type=float,
        help='fgsm epsilon')
parser.add_argument('--lr', default='1e-4', type=float,
        help='learning rate')
parser.add_argument('--epochs', default='2 0', type=int,
        help='number of epochs')
parser.add_argument('--vis', action='store_true',
        help='whether visualize result')
args = parser.parse_args()

torch.set_num_threads(6)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range

    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_dir, args.dataset)

    epsilon = args.epsilon

    # create model
    model = Steath([1, 3, 255, 255])
    track_model = ModelBuilder()
    lr = args.lr

    # load model
    model = load_pretrain(model, args.snapshot).cuda()
    track_model = load_pretrain(track_model, args.snapshot).cuda().eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    # model.dx.requires_grad_(True)
    # model.backbone.eval()
    # if cfg.ADJUST.ADJUST:
    #     model.neck.eval()
    # model.rpn_head.eval()

    for name, param in model.named_parameters():

        if 'backbone' in name or 'neck' in name or 'rpn_head' in name:
            param.requires_grad_(False)
        elif param.requires_grad:
            param.requires_grad_(True)
            print(name, param.data)
        else:
            print(name)

    clipper = WeightClipper(5)

    # build tracker
    tracker1 = build_tracker(track_model)
    tracker2 = build_tracker(track_model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            config=cfg)
    #
    # vid.name = {'ants1','ants3',....}
    # img, bbox, cls, delta, delta_weight
    # vid[0][0],vid[0][1],vid[0][2],vid[0][3],vid[0][4]

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    n_epochs = args.epochs

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:

        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
                else:
                    if not os.path.exists(os.path.join(args.savedir, video.name)):
                        os.mkdir(os.path.join(args.savedir, video.name))

            # set writing video parameters
            height, width, channels = video[0][0].shape
            out = cv2.VideoWriter(os.path.join(args.savedir, video.name + '.avi'),
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (width, height))
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            data = {'template': None, 'search': None}

            for idx, (img, gt_bbox, z, x, szx, boxx, padx, cls, delta, delta_w, overlap, _bbox, _bbox_p) in enumerate(video):

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]

                tic = cv2.getTickCount()

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-w//2, cy-h//2, w, h]

                if idx == frame_counter:
                    tracker1.init(img, gt_bbox_)
                    tracker2.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)

                    data['template'] = torch.autograd.Variable(z, requires_grad=True).cuda()

                elif idx > frame_counter:
                    prim_img = np.copy(img)
                    data['search'] = torch.autograd.Variable(x, requires_grad=True).cuda()
                    data['label_cls'] = torch.Tensor(cls).type(torch.LongTensor).cuda()
                    data['label_loc'] = torch.Tensor(delta).type(torch.FloatTensor).cuda()
                    data['label_loc_weight'] = torch.Tensor(delta_w).cuda()

                    diff = data['search']

                    for epoch in range(n_epochs):
                        outputs = model(data, epsilon)
                        cls_loss = outputs['cls_loss']
                        # print(idx, epoch, cls_loss.item())
                        loc_loss = outputs['loc_loss']
                        total_loss = outputs['total_loss']

                        print('{}/{} cls={}, loc={}, total={}'.format(idx, len(video), cls_loss.item(), loc_loss.item(),
                                                                      total_loss.item()))

                        optimizer.zero_grad()
                        # cls_loss.backward()
                        total_loss.backward()
                        # model.apply(clipper)
                        optimizer.step()

                        # print('loss ', loss(diff, outputs['search']).item())
                        # diff = outputs['search']

                    # print(epoch, cls_loss, loc_loss, total_loss)
                    # print('{}/{} cls={}, loc={}, total={}'.format(idx, len(video), cls_loss.item(), loc_loss.item(),
                    #                                               total_loss.item()))
                    perturb_data = outputs['search']

                    # cv2.rectangle(img, (int(cx-w/2+1), int(cy-h/2+1)), (int(cx+w/2+1), int(cy+h/2+1)), (0, 0, 0), 3)
                    # cv2.imwrite(os.path.join(args.savedir, video.name, 'original_' + str(idx).zfill(7) + '.jpg'), img)

                    # _img = perturb_data.data.cpu().numpy().squeeze().transpose([1, 2, 0])
                    # cv2.imwrite(os.path.join(args.savedir, 'perturb_' + str(idx) + '.jpg'), _img)

                    szx = int(szx)

                    if not np.array_equal(cfg.TRACK.INSTANCE_SIZE, szx):
                        perturb_data = F.interpolate(perturb_data, size=szx)
                        __bbox = (np.array(_bbox_p)*szx/cfg.TRACK.INSTANCE_SIZE).astype(np.int)

                    _img = cv2.UMat(perturb_data.data.cpu().numpy().squeeze().transpose([1, 2, 0])).get()
                    cv2.rectangle(_img, (__bbox[0], __bbox[1]), (__bbox[2], __bbox[3]), (0, 0, 0), 3)
                    cv2.imwrite(os.path.join(args.savedir, video.name, 'crop_full_' + str(idx) + '.jpg'), _img)

                    nh, nw, _ = _img.shape

                    __bbox0 = np.zeros_like(__bbox)
                    __bbox0[:4:2] = __bbox[:4:2] - padx[0]
                    __bbox0[1:4:2] = __bbox[1:4:2] - padx[2]

                    img[boxx[0]:boxx[1] + 1, boxx[2]:boxx[3] + 1, :] = \
                        _img[boxx[0]+padx[0]:boxx[1]+padx[0] + 1, 0 + padx[2]:boxx[3] - boxx[2] + padx[2] + 1, :]
                    # cv2.imwrite(os.path.join(args.savedir, video.name, 'perturb_full_' + str(idx) + '.jpg'), img)

                    # if not np.array_equal(cfg.TRACK.INSTANCE_SIZE, sz):
                    #     perturb_data = F.interpolate(perturb_data, size=sz)
                    #     __bbox = (np.array(_bbox)*sz/cfg.TRACK.INSTANCE_SIZE).astype(np.uint8)
                    #
                    # _img = cv2.UMat(perturb_data.data.cpu().numpy().squeeze().transpose([1, 2, 0])).get()
                    # cv2.rectangle(_img, (__bbox[0], __bbox[1]), (__bbox[2], __bbox[3]), (0, 0, 0), 3)
                    # cv2.imwrite(os.path.join(args.savedir, video.name, 'crop_full_' + str(idx) + '.jpg'), _img)
                    #
                    # nh, nw, _ = _img.shape
                    # img[bT:bB+1, bL:bR+1, :] = _img[pad[0]:nh - pad[1], pad[2]:nw - pad[3], :]
                    # cv2.imwrite(os.path.join(args.savedir, video.name, 'perturb_full_' + str(idx) + '.jpg'), img)

                    # nimg, sz, box, pad = tracker2.crop(img, bbox=gt_bbox_, im_name='search' + str(idx))

                    outputs = tracker1.track(img)
                    prim_outputs = tracker2.track(prim_img)

                    pred_bbox = outputs['bbox']
                    prim_box = prim_outputs['bbox']

                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)

                # cv2.imwrite(os.path.join(args.savedir, video.name, str(idx).zfill(7) + '.jpg'), img)

                toc += cv2.getTickCount() - tic

                # write ground truth bbox
                cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                              True, (255, 255, 255), 3)

                if idx != frame_counter:
                    bbox = list(map(int, pred_bbox))
                    prim_bbox = list(map(int, prim_box))

                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)

                    cv2.rectangle(img, (prim_bbox[0], prim_bbox[1]),
                                  (prim_bbox[0] + prim_bbox[2], prim_bbox[1] + prim_bbox[3]), (0, 0, 255), 3)


                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                out.write(img)
                cv2.imwrite(os.path.join(args.savedir, video.name, str(idx).zfill(7) + '.jpg'), img)

                # import pdb
                # pdb.set_trace()

            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
