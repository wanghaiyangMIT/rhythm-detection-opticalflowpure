from vgg_net import Vgg16Conv
import argparse
import torch
from dataloader import *
from torch.utils.data import DataLoader
import cv2
import json
import os
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"------------------------------dataloader------------------------------"
parser.add_argument('--videodir',dest='inputpath',type = str,help='video-directory',default="")
parser.add_argument('--time',dest = 'time',default=4500,type=int,help='choose the unify video length to train and test')
parser.add_argument('--inpdim',dest = 'inp_dim',default=224,type=int,help = 'input dim for vgg19')
parser.add_argument('--efps',dest = 'efps',default=20,type=int,help='the count of extract frame per second')
parser.add_argument('--indim',dest = 'indim',default=224,type=int,help='imgsize input vgg')
parser.add_argument('--outdir',dest = 'outdir',default="",type=str,help='save the feature generate by vgg')
opt = parser.parse_args()

rootdir = os.getcwd()
if not os.path.exists(opt.outdir):
    os.mkdir(opt.outdir)
    
if __name__ == '__main__':

    #load video data
    video_dir = opt.inputpath
    vggdataset = RawVideoDataLoader(video_dir)
    vggdataloader = DataLoader(vggdataset,pin_memory = True)
    model = Vgg16Conv().cuda()
    for video_data ,video_name in tqdm(vggdataloader):
        if ((video_data == False)or (video_name == False)):
            continue
        vggfeature_map = []
        time = []
        for i in range(len(video_data['time_list'])):
            timerecord = video_data['time_list'][i].numpy().tolist()
            time.append(timerecord)
        for perframe in tqdm(video_data['crop_frame']):
            perframe = perframe.cuda()
            output = model(perframe)
            output = output.view(1000,1) #numclass = 1000
            output = output.cpu()
            output = output.detach().numpy().tolist()
            vggfeature_map.append(output)
        vggfeature_map = np.array(vggfeature_map)
        vggfeature_map = vggfeature_map.transpose(1,0,2).tolist()    
        dict_result = {'vggfeaturemap':vggfeature_map,'time':time}
        json_result = json.dumps(dict_result)
        print(os.path.join(opt.outdir,str(video_name[0][:-4])+'_vggfeaturemap.json'))
        with open (os.path.join(opt.outdir,str(video_name[0][:-4])+'_vggfeaturemap.json'),'w') as f:
            f.write(json_result)
        
