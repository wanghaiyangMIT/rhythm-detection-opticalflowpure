import argparse
import json
import os
import random
from tqdm import tqdm
import time

import cv2
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader#, Subset
import models
from data_loader import get_dataset
from utils import collate_fn, DataError

parser = argparse.ArgumentParser(description='PyTorch Training')
rootdir = os.getcwd()
# environment
parser.add_argument('--audio_dir',  type=str,   default='')
parser.add_argument('--video_dir',  type=str,   default='')
# parser.add_argument('--vggf_dir',   type=str,   default='')
parser.add_argument('--save_dir',   type=str,   default='')
parser.add_argument('--debug',      type=int,   default=0,      help='use a small test set')
parser.add_argument('--device',     type=int,   default=-1,     help='gpu device id')
# parser.add_argument('--infer_answer', type=int, default=0)
# train
parser.add_argument('--epoch_size', type=int,   default=200,    help='how many segments in an epoch')
parser.add_argument('--batch_size', type=int,   default=1,      help='batch size')
parser.add_argument('--num_epoch',  type=int,   default=10000,  help='maximum number of epochs')
parser.add_argument('--eval_every', type=int,   default=1,      help='evaluate how many every epoch')
parser.add_argument('--save_every', type=int,   default=100,    help='save model how many every epoch')
parser.add_argument('--max_f1',     type=float, default=0.0,    help='save if f1 exceed max_f1')
parser.add_argument('--lr',         type=float, default=0.00003)
# data
parser.add_argument('--fps',        type=int,   default=4)
parser.add_argument('--use_label',  type=int,   default=1,      help='use 0/1 label as target instead of real number in [0,1]')
parser.add_argument('--theta',      type=float, default=0.3,    help='onset threshold')
parser.add_argument('--num_sample', type=int,   default=4,      help='sample how many segments in a video during one epoch')
# parser.add_argument('--mask', type=int, default=0, help='mask some negative data points')
# parser.add_argument('--label', type=str, default='strength')
# model
parser.add_argument('--segment_length', type=int, default=20)
parser.add_argument('--dim_feature',    type=int, default=1000, help='dim of vgg feature')
parser.add_argument('--dim_video',      type=int, default=224,  help='H and W of video frames')
parser.add_argument('--model',          type=str, default='EndToEnd')
parser.add_argument('--use_crf',        type=int, default=1,    help='whether to use CRF layer to predict, must use 0/1 label')
parser.add_argument('--vgg_init',       type=int, default=1,    help='whether to initialize the VGG net with pre-trained params')
#parser.add_argument('--shift_with_attention', type=int, default=0)

args = parser.parse_args()
if len(args.save_dir) == 0:
    if args.debug:
        args.save_dir = 'debug'
    else:
        args.save_dir = 'train'
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
else:
    print('save_dir \'%s\' already exist, input YES to comfirm' % args.save_dir)
    s = input()
    if 'YES' not in s:
        assert False
if args.use_crf and not args.use_label:
    print('the target of CRF must be 0/1 label')
    assert False

######################################################

def handle_batch(args, batch):
    video, label = batch
    if len(label.shape) == 1:
        raise DataError('handle_batch: bad batch')
    with torch.no_grad():
        video = video.cuda()    # (n, T, H, W)
        label = label.cuda()    # (n, T, 1) long
    pred = model(video)         # (n, T, 1) long
    return video, label, pred

def train(model, train_set, test_set, args):
    data_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn)

    adam = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    for epoch in range(args.num_epoch):
        model.train()
        epoch_loss = 0
        epoch_trained = 0
        
        for batch in tqdm(data_loader):
            try:
                video, label, pred = handle_batch(args, batch)
                #print(type(pred))
                adam.zero_grad()
                loss = model.loss(video, label)
                loss.backward()
                adam.step()
                epoch_loss += loss.item()
                epoch_trained += 1
                for i in model.Resnet50.layer4.parameters():
                    print(torch.sum(i.grad))
                for j in model.Resnet50.layer1.parameters():
                    print(torch.sum(j.grad))
            except DataError as e:
                # print(e)
                continue
        print(pred.reshape(-1))
        print(label.reshape(-1))     
        report = 'epoch[%d/%d] Loss: %.5f' % (epoch+1, args.num_epoch, epoch_loss / max(1, epoch_trained))
        with open (os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
            f.write(report + '\n')
        print(report)

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_%i_params.pkl' % epoch))
        if epoch % args.eval_every == 0:
            evaluate(model, test_set, args)

def evaluate(model, test_set, args):
    data_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn)
    cnt_match, cnt_label, cnt_pred, cnt_length = 0, 0, 0, 0
    model.eval()
    for batch in tqdm(data_loader):
        try:
            video, label, pred = handle_batch(args, batch)
            cnt_label  += torch.sum(label).item()
            cnt_pred   += torch.sum(pred).item()
            cnt_match  += torch.sum(label.mul(pred.byte())).item()
            cnt_length += label.shape[0] * label.shape[1]
            # for n in range(label.shape[0]):
            #     for t in range(label.shape[1]):
            #         flag_label = True if label[n,t,0] > args.theta else False
            #         flag_pred = True if pred[n,t,0] > args.theta else False
            #         cnt_label += flag_label
            #         cnt_pred += flag_pred
            #         cnt_match += flag_label and flag_pred
        except:
            # print(e)
            continue    
    print(pred.reshape(-1))
    print(label.reshape(-1))

    prec   = 1. * cnt_match / cnt_pred if cnt_pred > 0 else 0
    recall = 1. * cnt_match / cnt_label
    f1 = 2. * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    report = 'Evaluation: F1 %.4f (%.4f %i/%i, %.4f %i/%i, %i)' % \
        (f1, prec, cnt_match, cnt_pred, recall, cnt_match, cnt_label, cnt_length)
    with open (os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
        f.write(report + '\n')
    print(report)
    if f1 > args.max_f1:
        args.max_f1 = f1
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'f1_%.4f_params.pkl' % f1))
if __name__ == '__main__':

    # prepare dataseut
    train_set, test_set = get_dataset(args)

    # define model
    Model = getattr(models, args.model)
    if args.device > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
        model = Model(args).cuda()

    # record
    localtime = time.asctime(time.localtime(time.time()))
    with open (os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
        f.write('*********** %s ***********\n' % localtime)
    with open(os.path.join(args.save_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)

    # train
    train(model, train_set, test_set, args)
