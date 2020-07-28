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
from pysot.models.model_builder_oneshot import ModelBuilder
from pysot.models.model_attack import ModelAttacker
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from tqdm import tqdm

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


def save(img, imga, szx, boxx, pad, filename, save=False):
    if save:
        szx = int(szx)
        imga2 = F.interpolate(imga, size=szx)
        # boxx = (np.array(boxx) * szx / cfg.TRACK.EXEMPLAR_SIZE).astype(np.int)

        imga2 = cv2.UMat(imga2.data.cpu().numpy().squeeze().transpose([1, 2, 0])).get()
        # cv2.rectangle(imga, (bboxa[0], bboxa[1]), (bboxa[2], bboxa[3]), (0, 0, 0), 3)
        boxx = np.array(boxx).astype(np.int)
        L = int(boxx[0] + (boxx[2] + 1) / 2 - (imga2.shape[0] + 1) / 2) - 1
        R = int(boxx[0] + (boxx[2] + 1) / 2 + (imga2.shape[0] + 1) / 2) - 1
        T = int(boxx[1] + (boxx[3] + 1) / 2 - (imga2.shape[0] + 1) / 2) - 1
        B = int(boxx[1] + (boxx[3] + 1) / 2 + (imga2.shape[0] + 1) / 2) - 1

        if boxx[2] % 2 == 1:
            L -= 1
            R -= 1
        if boxx[3] % 2 == 1:
            T -= 1
            B -= 1

        if T < 0:
            B -= T
            T = 0

        if L < 0:
            R -= L
            L = 0

        if B - T - 1 != imga2.shape[1] or R - L - 1 != imga2.shape[1]:
            pdb.set_trace()

        imgn = img.copy()
        imgn[T:B - 1, L:R - 1, :] = imga2
        cv2.imwrite(filename, imgn)
        # imgx = imgn[L:R-1, T:B-1, :]
        # sum(sum(imga2 - imgx))

    return imgn


def save_2bb(imgX, filename, ad_bbox, pred_bbox, gt_bbox):
    img = imgX.copy()

    # __gt_bbox = list(map(int, gt_bbox_))
    bbox = list(map(int, pred_bbox))

    ad_bbox = list(map(int, ad_bbox))
    cv2.rectangle(img, (ad_bbox[0], ad_bbox[1]),
                  (ad_bbox[0] + ad_bbox[2], ad_bbox[1] + ad_bbox[3]), (0, 0, 255), 3)

    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)

    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                  True, (0, 0, 0), 3)

    # cv2.rectangle(img, (__gt_bbox[0], __gt_bbox[1]),
    #               (__gt_bbox[0] + __gt_bbox[2], __gt_bbox[1] + __gt_bbox[3]), (0, 0, 0), 3)

    cv2.imwrite(filename, img)


def stoa_track(idx, frame_counter, img, gt_bbox, tracker1):
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    lost_number = 0

    if idx == frame_counter:
        init_gt = gt_bbox_
        tracker1.init(img, gt_bbox_)
        # pred_bboxes.append(1)
        append = 1
    elif idx > frame_counter:

        outputs = tracker1.track(img, idx=idx)

        # print('****************** state of the art tracking ******************')
        append = outputs['bbox']

        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
        if args.dataset != 'OTB100':
            if overlap > 0:
                # not lost
                lost = False
            else:
                # lost object
                append = 2
                frame_counter = idx + 5  # skip 5 frames
                lost_number = 1
                lost = True
        else:
            if overlap <= 0:
                lost_number = 1

    else:
        append = 0

    return append, lost_number, frame_counter


def adversarial_train(idx, frame_counter, state, attacker, tracker, optimizer, pbar):
    img = state['img']
    gt_bbox = state['gt_bbox']

    cx, cy, w, h = get_axis_aligned_bbox(np.array(state['gt_bbox']))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    lost_number = 0

    state['zimg'] = img.copy()
    state['init_gt'] = gt_bbox_

    state['sz'], state['bbox'], state['pad'] = \
        tracker.init(img, gt_bbox_, attacker=attacker, epsilon=args.epsilon)

    for iter in range(0, 100):
        _outputs = tracker.train(img, attacker=attacker, epsilon=args.epsilon, iter=iter)

        l1 = _outputs['l1']
        l2 = _outputs['l2']

        total_loss = l1 + 0.4*l2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(iter, total_loss.item(), l1.item(), l2.item())

        save(state['zimg'], tracker.z_crop_adv, state['sz'], state['init_gt'], state['pad'],
            os.path.join(args.savedir, state['video_name'], str(idx).zfill(6) + '.jpg'), save=True)

    # return append, lost_number, frame_counter
    return optimizer, state, _outputs['bbox'] if idx > 0 else 0


def main():
    mode = 0
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_dir, args.dataset)

    epsilon = args.epsilon

    # create model
    track_model = ModelBuilder()
    track_model0 = ModelBuilder()
    lr = args.lr

    # load model
    track_model = load_pretrain(track_model, args.snapshot).cuda().eval()
    track_model0 = load_pretrain(track_model0, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(track_model)
    tracker0 = build_tracker(track_model0)

    attacker = ModelAttacker().cuda().train()
    optimizer = optim.Adam(attacker.parameters(), lr=lr)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            dataset_toolkit='oneshot',
                                            config=cfg)
    #
    # vid.name = {'ants1','ants3',....}
    # img, bbox, cls, delta, delta_weight
    # vid[0][0],vid[0][1],vid[0][2],vid[0][3],vid[0][4]

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    n_epochs = args.epochs

    for name, param in tracker.model.named_parameters():
        param.requires_grad_(False)

    # for name, param in attacker.named_parameters():
    #     if 'backbone' in name or 'neck' in name or 'rpn_head' in name:
    #         param.requires_grad_(False)

    # for name, param in tracker2.model.named_parameters():
    #     if 'backbone' in name or 'neck' in name or 'rpn_head' in name:
    #         param.requires_grad_(False)
    #     elif param.requires_grad:
    #         param.requires_grad_(True)
    #         # print(name, param.data)
    #         print('grad true ', name)
    #     else:
    #         print('grad false ', name)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019', 'OTB100']:

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
            frame_counter_adv = 0
            lost_number = 0
            lost_number_adv = 0
            toc = 0
            pred_bboxes = []
            pred_bboxes_adv = []

            pbar = tqdm(enumerate(video))
            for idx, (img, gt_bbox) in pbar:

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                ##########################################
                # # #  for state of the art tracking # # #
                ##########################################
                if mode == 0:
                    pred_bbox, _lost, frame_counter = stoa_track(idx, frame_counter, img, gt_bbox, tracker0)

                tic = cv2.getTickCount()
                ##########################################
                # # # # #  adversarial tracking  # # # # #
                ##########################################
                if idx == 0:
                    state = {
                        'img': img,
                        'gt_bbox': gt_bbox,
                        'video_name': video.name
                    }
                else:
                    state['img'] = img

                if mode == 1:
                    optimizer, state, ad_bbox = \
                        adversarial_train(idx, frame_counter_adv, state, attacker, tracker, optimizer, pbar)
                    if idx == 0:
                        break

                toc += cv2.getTickCount() - tic

                if idx > 0 and mode == 0:
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)

                if idx > 0 and mode == 1:
                    ad_bbox = list(map(int, ad_bbox))
                    cv2.rectangle(img, (ad_bbox[0], ad_bbox[1]),
                                  (ad_bbox[0] + ad_bbox[2], ad_bbox[1] + ad_bbox[3]), (0, 0, 255), 3)

                __gt_bbox = list(map(int, gt_bbox_))
                cv2.rectangle(img, (__gt_bbox[0], __gt_bbox[1]),
                              (__gt_bbox[0] + __gt_bbox[2], __gt_bbox[1] + __gt_bbox[3]), (0, 0, 0), 3)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                out.write(img)

            toc /= cv2.getTickFrequency()

            # save results
            if args.dataset == 'OTB100':
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes_adv:
                        f.write(','.join([str(i) for i in x]) + '\n')
            else:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))

                # ii = 0
                # with open(result_path, 'r') as f:
                #     xs = f.readlines()
                #     for x in xs:
                #         if ii == 0:
                #             pred_bboxes_adv[0] = ','.join([vot_float2str("%.4f", i) for i in pred_bboxes_adv[0]]) + '\n'
                #         else:
                #             pred_bboxes_adv.append(x)
                #         ii += 1
                #
                # with open(result_path, 'w') as f:
                #     for x in pred_bboxes_adv:
                #         f.write(x)

                with open(result_path, 'w') as f:
                    for x in pred_bboxes_adv:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number_adv))
            total_lost += lost_number_adv
        print("{:s} total lost: {:d}".format(model_name, total_lost))


if __name__ == '__main__':
    main()
