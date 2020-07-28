from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import logging
import random

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

from torch.utils.data import DataLoader
from pysot.datasets.two_pass_dataset import TwoPassDataset

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


def seed_train(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(train_dataset):
    train_sampler = None
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


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


def stoa_track(idx, frame_counter, img, gt_bbox, tracker):
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    lost_number = 0

    if idx == frame_counter:
        tracker.init(img, gt_bbox_)
        append = 1
    elif idx > frame_counter:

        outputs = tracker.track(img, idx=idx)

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

    return append, lost_number, frame_counter, tracker


def adversarial_track(idx, frame_counter, img, gt_bbox, attacker, tracker, optimizer):
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    lost_number = 0

    if idx == frame_counter:
        zimg = img.copy()
        tracker.init(img, gt_bbox_, attacker=attacker, epsilon=args.epsilon)

        zf2 = tracker.zf

        if args.dataset == 'OTB100':
            append = gt_bbox_
        else:
            append = 1
        # cv2.imwrite(os.path.join(args.savedir, video.name, str(idx).zfill(6) +'.jpg'), img)

    elif idx > frame_counter:
        _outputs = tracker.track(img, attacker=attacker, epsilon=args.epsilon, idx=idx, iter=i)
        # update state
        tracker.center_pos = _outputs['center_pos']
        tracker.size = _outputs['size']

        ad_overlap = vot_overlap(_outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
        append = _outputs['bbox']

        if ad_overlap <= 0:
            lost_number = 1
            if args.dataset != 'OTB100':
                # lost object
                append = 2
                frame_counter = idx + 5  # skip 5 frames
    else:
        append = 0

    return append, lost_number, frame_counter, tracker, optimizer


def first_pass(z, x, attacker, tracker, optimizer):
    tracker.first_pass(z, x)


def main():

    seed_train()
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_dir, args.dataset)

    epsilon = args.epsilon

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            dataset_toolkit='oneshot',
                                            config=cfg)

    # train_dataset = TwoPassDataset()

    # build dataset loader
    # train_loader = build_data_loader(dataset)

    # create model
    track_model = ModelBuilder()
    lr = args.lr

    # load model
    track_model = load_pretrain(track_model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(track_model)

    attacker = ModelAttacker().cuda().train()
    optimizer = optim.Adam(attacker.parameters(), lr=lr)

    #
    # vid.name = {'ants1','ants3',....}
    # img, bbox, cls, delta, delta_weight
    # vid[0][0],vid[0][1],vid[0][2],vid[0][3],vid[0][4]

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    n_epochs = args.epochs

    for name, param in tracker.model.named_parameters():
        param.requires_grad_(False)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019', 'OTB100']:

        # restart tracking
        for v_idx, video in enumerate(dataset):
            pdb.set_trace()
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

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

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

                pred_bbox, _lost, frame_counter, tracker = stoa_track(idx, frame_counter, img, gt_bbox, tracker)
                pred_bboxes.append(pred_bbox)
                lost_number += _lost

                end_t = 0
                tic = cv2.getTickCount()
                ##########################################
                # # # # #  adversarial tracking  # # # # #
                ##########################################
                skip = False

                if not skip:
                    # ad_bbox, _lost_adv, frame_counter_adv, tracker2, optimizer = \
                    #     adversarial_track(idx, frame_counter_adv, img, gt_bbox, attacker, tracker2, optimizer)
                    z = {
                        'img': img,
                        'bbox': gt_bbox
                    }

                    first_pass(z, x, attacker, optimizer)
                lost_number_adv += _lost_adv
                pred_bboxes_adv.append(ad_bbox)

                toc += cv2.getTickCount() - tic

                # pdb.set_trace()

                if idx > frame_counter_adv and _lost_adv > 0:
                    ad_bbox = list(map(int, ad_bbox))
                    cv2.rectangle(img, (ad_bbox[0], ad_bbox[1]),
                                  (ad_bbox[0] + ad_bbox[2], ad_bbox[1] + ad_bbox[3]), (0, 0, 255), 3)

                if idx > frame_counter and not _lost > 0:
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)

                __gt_bbox = list(map(int, gt_bbox_))
                cv2.rectangle(img, (__gt_bbox[0], __gt_bbox[1]),
                              (__gt_bbox[0] + __gt_bbox[2], __gt_bbox[1] + __gt_bbox[3]), (0, 0, 0), 3)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, "," + str(lost_number_adv), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                out.write(img)

                # print('frame {}/{} -> {} epochs'.format(idx, len(video), end_t))

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