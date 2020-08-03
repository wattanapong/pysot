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
import math
import pdb
from torchvision.transforms import ToTensor, ToPILImage

from pysot.core.config import cfg
from pysot.models.model_builder_oneshot import ModelBuilder
from pysot.models.model_attack_2pass import ModelAttacker
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox, get_axis_aligned_bbox_tensor
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from tqdm import tqdm
from pysot.datasets.myDataset import MyDataset

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
parser.add_argument('--mode', default='train', type=str,
                    help='train or test mode')
parser.add_argument('--batch', default=16, type=int,
                    help='batch size')
parser.add_argument('--lr', default='1e-4', type=float,
                    help='learning rate')
parser.add_argument('--alpha', default='0.7', type=float,
                    help='alpha')
parser.add_argument('--beta', default='0.3', type=float,
                    help='beta')
parser.add_argument('--gamma', default='0.1', type=float,
                    help='gamma')
parser.add_argument('--epochs', default='2 0', type=int,
                    help='number of epochs')
parser.add_argument('--video_idx', default=0, type=int,
                    help='start video from idx')
parser.add_argument('--vis', action='store_true',
                    help='whether visualize result')
parser.add_argument('--debug', action='store_true',
                    help='debugging flag')
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

        imgn = img.copy()

        bb1 = [T, B - 1, L, R - 1]
        bb2 = [0, imgn.shape[0], 0, imgn.shape[1]]
        flag = False

        if T < 0:
            bb1[:2] = [0, B - 1]
            bb2[:2] = [-T, B - T - 1]
            flag = True
        elif B > imgn.shape[0]:
            bb1[:2] = [T, imgn.shape[0]]
            bb2[:2] = [0, imga2.shape[0] - (B - imgn.shape[0]) + 1]
            flag = True
        if L < 0:
            bb1[2:4] = [0, R - 1]
            bb2[2:4] = [-L, R - L - 1]
            flag = True
        elif R > imgn.shape[1]:
            bb1[2:4] = [L, imgn.shape[1]]
            bb2[2:4] = [0, imga2.shape[1] - (R - imgn.shape[1] - 1)]
            flag = True

        if flag:
            # pdb.set_trace()
            imgn[bb1[0]:bb1[1], bb1[2]:bb1[3], :] = imga2[bb2[0]:bb2[1], bb2[2]:bb2[3], :]
        else:
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


def adversarial_train(idx, state, attacker, tracker, optimizer, gt_bbox, epoch):
    img = state['img']
    cx, cy, w, h = gt_bbox
    cx, cy = cx + w // 2, cy + h // 2

    if idx == 0:
        state['zimg'] = img.copy()
        state['init_gt'] = gt_bbox
        state['sz'], state['bbox'], state['pad'] = \
            tracker.init(state['zimg'], state['init_gt'], attacker=attacker, epsilon=args.epsilon, update=True)
    if idx > 0:
        tracker.init(state['zimg'], state['init_gt'], attacker=attacker, epsilon=args.epsilon, update=False)
        _outputs = tracker.train(img, attacker=attacker, bbox=torch.stack([cx, cy, w, h]), epsilon=args.epsilon,
                                 batch=args.batch, idx=idx)

        l1 = _outputs['l1']
        l2 = _outputs['l2']
        # l3 = _outputs['l3']

        # if idx == 1:
        #     total_loss = args.alpha * l1 + args.beta * l2
        # else:
        #     total_loss = args.alpha * l1 + args.beta * l2 + args.gamma * l3

        total_loss = args.alpha * l1 + args.beta * l2

        optimizer.zero_grad()
        total_loss.sum().backward()
        optimizer.step()
        # with torch.no_grad():
        #     attacker.adv_z[attacker.adv_z != attacker.adv_z] = 0

        # save(state['zimg'], attacker.perturb(tracker.z_crop, args.epsilon), state['sz'], state['init_gt'], state['pad'],
        #     os.path.join(args.savedir, state['video_name'], str(idx).zfill(6) + '.jpg'), save=True)

    # return state, [total_loss.item(), l1.item(), l2.item(), l3.item() if epoch > 0 else 0] if idx > 0 else 0
    return state, [total_loss.sum().item(), l1.sum().item(), l2.sum().item()] if idx > 0 else 0


def main():
    mode = args.mode
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

    # vid.name = {'ants1','ants3',....}
    # img, bbox, cls, delta, delta_weight
    # vid[0][0],vid[0][1],vid[0][2],vid[0][3],vid[0][4]

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    n_epochs = args.epochs

    for name, param in tracker.model.named_parameters():
        param.requires_grad_(False)

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

            elif v_idx < args.video_idx and args.debug:
                continue

            # set writing video parameters
            height, width, channels = video[0][0].shape
            if mode == 'test':
                out = cv2.VideoWriter(os.path.join(args.savedir, video.name + '.avi'),
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (width, height))
            frame_counter = 0
            toc = 0
            pred_bboxes_adv = []
            adv_z = []

            track_model = load_pretrain(track_model, args.snapshot).cuda().eval()

            # build tracker
            tracker = build_tracker(track_model)

            attacker = ModelAttacker().cuda().train()
            # optimizer = optim.Adam(attacker.parameters(), lr=lr)
            optimizer = optim.SGD(attacker.parameters(), lr=lr, momentum=0.9)

            # load pretrained
            start_epoch = 0
            if os.path.exists(os.path.join(args.savedir, 'checkpoint', video.name)):
                entries = os.listdir(os.path.join(args.savedir, 'checkpoint', video.name))
                if len(entries) > 0:
                    entries.sort()
                    start_epoch = int(entries[-1][-7:-4]) if entries[-1][-7:-4].isnumeric() else 0

            if start_epoch > 0:
                state = torch.load(os.path.join(args.savedir, 'checkpoint', video.name, entries[-1]))
                attacker.load_state_dict(state['attacker'])
                optimizer.load_state_dict(state['optimizer'])

            # generate cropping offset
            tracker.generate_transition(64, len(video))

            training_data = MyDataset()
            num_frames = len(video)

            iter = math.ceil((num_frames-1) / args.batch)
            params = {'batch_size': args.batch,
                      'shuffle': False,
                      'num_workers': 6}
            for i in range(0, iter-1):
                for j in range(0, args.batch):
                    training_data.add([video[i*args.batch+j][0], video[i*args.batch+j][1]])

            for j in range(0, num_frames - 1 - (iter-1)*args.batch):
                training_data.add([video[(iter-1) * args.batch + j][0], video[(iter-1) * args.batch + j][1]])
            data_loader = torch.utils.data.DataLoader(training_data, **params)

            for epoch in range(0, args.epochs):

                ##########################################
                # # #  for state of the art tracking # # #
                ##########################################
                if mode == 'test':
                    if epoch > 0:
                        break

                    pbar = tqdm(enumerate(video), position=0, leave=True)

                    for idx, (img, gt_bbox) in pbar:

                        if len(gt_bbox) == 4:
                            gt_bbox = [gt_bbox[0], gt_bbox[1],
                                       gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]

                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                        tic = cv2.getTickCount()
                        pred_bbox, _lost, frame_counter = stoa_track(idx, frame_counter, img, gt_bbox, tracker0)
                        toc += cv2.getTickCount() - tic

                        if idx > 0:
                            bbox = list(map(int, pred_bbox))
                            cv2.rectangle(img, (bbox[0], bbox[1]),
                                          (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)

                        # if idx > 0 and mode == 1:
                        #     ad_bbox = list(map(int, ad_bbox))
                        #     cv2.rectangle(img, (ad_bbox[0], ad_bbox[1]),
                        #                   (ad_bbox[0] + ad_bbox[2], ad_bbox[1] + ad_bbox[3]), (0, 0, 255), 3)

                        __gt_bbox = list(map(int, gt_bbox_))
                        cv2.rectangle(img, (__gt_bbox[0], __gt_bbox[1]),
                                      (__gt_bbox[0] + __gt_bbox[2], __gt_bbox[1] + __gt_bbox[3]), (0, 0, 0), 3)

                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                        out.write(img)

                elif mode == 'train':

                    if epoch < start_epoch:
                        continue

                    # initial frame
                    img, gt_bbox = video[0]
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                                   gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    state = {
                        'img': img,
                        'gt_bbox': gt_bbox_,
                        'video_name': video.name
                    }

                    state, loss = adversarial_train(0, state, attacker, tracker, optimizer, gt_bbox_, epoch)
                    pbar = tqdm(enumerate(data_loader), position=0, leave=True)
                    _loss = []

                    for (_idx, (idx, (imgs, gt_bboxes))) in enumerate(pbar):
                        if len(gt_bboxes[0]) == 4:
                            gt_bboxes = (gt_bboxes[:, 0], gt_bboxes[:, 1],
                                       gt_bboxes[:, 0], gt_bboxes[:, 1] + gt_bboxes[:, 3] - 1,
                                       gt_bboxes[:, 0] + gt_bboxes[:, 2] - 1, gt_bboxes[:, 1] + gt_bboxes[:, 3] - 1,
                                       gt_bboxes[:, 0] + gt_bboxes[:, 2] - 1, gt_bboxes[:, 1])

                        gt_bboxes = torch.stack(gt_bboxes).float()
                        cx, cy, w, h = get_axis_aligned_bbox_tensor(gt_bboxes)
                        gt_bboxes_ = torch.stack([cx, cy, w, h])

                        tic = cv2.getTickCount()

                        ##########################################
                        # # # # #  adversarial tracking  # # # # #
                        ##########################################
                        state['img'] = imgs

                        state, loss = adversarial_train(args.batch * idx + 1, state, attacker, tracker, optimizer,
                                                        gt_bboxes_, epoch)
                        if idx > 0:
                            _loss.append(loss)
                            # pbar.set_postfix_str('%d. Video: %s epoch: %d total %.3f %.3f %.3f %.3f %.3f' %
                            #                      (v_idx + 1, video.name, epoch + 1, loss[0], loss[1], loss[2], loss[3],
                            #                       attacker.adv_z.mean()))
                            pbar.set_postfix_str('%d. Video: %s epoch: %d ' % (v_idx + 1, video.name, epoch + 1))
                            # pbar.set_postfix_str('%d. Video: %s epoch: %d total %.3f %.3f %.3f %.3f' %
                            #                      (v_idx + 1, video.name, epoch + 1, loss[0], loss[1], loss[2], loss[3]))
                            adv_z.append(attacker.adv_z.data.cpu())

                        toc += cv2.getTickCount() - tic

                toc /= cv2.getTickFrequency()

                if mode == 'train':
                    # attacker.template_average = sum(adv_z) / len(adv_z)
                    # attacker.template_average[attacker.template_average != attacker.template_average] = 0
                    # attacker.template_average = torch.clamp(attacker.template_average.data, min=0, max=1)

                    # z_adv = attacker.add_noise(tracker.z_crop, attacker.template_average.cuda(), epsilon)
                    z_adv = attacker.add_noise(tracker.z_crop, attacker.adv_z, epsilon)

                    if not os.path.exists(os.path.join(args.savedir, state['video_name'])):
                        os.mkdir(os.path.join(args.savedir, state['video_name']))
                    save(state['zimg'], z_adv, state['sz'], state['init_gt'], state['pad'],
                         os.path.join(args.savedir, state['video_name'], str(epoch).zfill(6) + '.jpg'), save=True)

                    _loss = np.asarray(_loss)

                    _loss_v = sum(_loss, 0) / _loss.shape[0]
                    pbar.clear()
                    print('%d. Video: %s Time: %.2fs  epoch: %d total %.3f %.3f %.3f' %
                          (v_idx + 1, video.name, toc, epoch + 1, _loss_v[0], _loss_v[1], _loss_v[2]))

                    # save state dict
                    state_dict = {
                        'attacker': attacker.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch+1
                    }

                    checkpoint_path = os.path.join(args.savedir, 'checkpoint', video.name)
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    torch.save(state_dict, os.path.join(checkpoint_path, 'checkpoint_'+str(epoch+1).zfill(3)+'.pth'))

            if mode == 'test':
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

            #     print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            #         v_idx + 1, video.name, toc, idx / toc, lost_number_adv))


if __name__ == '__main__':
    main()
