import os
import pickle
import numpy as np

import cv2
import torch
from PIL import Image, ImageDraw
from utils import DataError

def letterbox_image(img, dim_video):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = dim_video, dim_video
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((dim_video, dim_video, 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas

def extract_frame(video_path, new_fps, dim_video):
    '''
    dim_video:      H and W
    beg and end:    in terms of frame in new_fps
    '''
    video_stream = cv2.VideoCapture(video_path)
    ori_len = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    ori_fps = video_stream.get(cv2.CAP_PROP_FPS)
    # print(ori_len, ori_fps)

    # video_stream.set(cv2.CAP_PROP_FPS, 4)
    # new_len = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # new_fps = video_stream.get(cv2.CAP_PROP_FPS)
    # print(new_len, new_fps)

    if ori_fps < new_fps:
        raise DataError('ori_fps < new_fps')
    if ori_len < 100 or ori_len > 10000:
        raise DataError('ori_len too short or too long')

    time_per_ori_frame = 1. / ori_fps
    time_per_new_frame = 1. / new_fps

    crop_list = []
    time_list = []
    time = 0    # real time
    for i in range(ori_len):
        (grabbed, frame) = video_stream.read()
        beg = i * time_per_ori_frame
        end = (i + 1) * time_per_ori_frame
        if time >= beg and time < end:
            frame = letterbox_image(frame, dim_video)
            crop_list.append(torch.tensor(frame).float())
            time_list.append(time)
            time = time + time_per_new_frame
        if time >= 80:
            break
    video_stream.release()

    result_dict = {}
    result_dict['crop_frame'] = crop_list
    result_dict['time_list'] = time_list
    return result_dict

if __name__ == '__main__':
    video_dir = '../video_3'
    video_list = os.listdir(video_dir)
    target_dir = '../video_3_frames_4fps'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    fps = 4
    dim_video = 224

    for i, video_name in enumerate(video_list):
        if i <= 1655:
            continue
        try:
            video_id = video_name.split('.')[0]
            group = video_id.split('_')[0]
            identi = video_id.split('_')[1]
        except:
            continue
        print('i identi: ', i, identi)

        try:
            video_time_dict = extract_frame(
                os.path.join(video_dir, video_name), fps, dim_video)
            video = torch.stack(video_time_dict['crop_frame'], dim=0).float()   # (T, H, W, 3)

            file_name = 'frames_' + str(identi) + '.pkl'
            with open(os.path.join(target_dir, file_name), 'wb') as f:
                pickle.dump(video, f)

        except Exception as e:
            # print(e)
            continue