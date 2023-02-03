
import argparse, cv2, torch, json
from pickle import TRUE
import numpy as np
import copy
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists
from torch.autograd import Variable

import time

from network.SiamR3D import *

from data_process.utils import rect_2_cxy_wh, cxy_wh_2_rect,get_template_z,\
    get_template_x1,_postprocess_score,_postprocess_box,xyxy2cxywh,overlap_ratio,xywh2xyxy
from run_Siam3D import *
from data_process.gauss import gaussian_label_function

parser = argparse.ArgumentParser(description='PyTorch SiamR3D OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB2015', help='datasets')



class track_config(object):
    numScale = 3
    scaleStep = 1.0375
    scalePenalty = 0.9745
    scaleLR = 0.59
    responseUp = 16
    wInfluence = 0.176
    z_lr = 0.1
    scale_min = 0.2
    scale_max = 5
    scale_resize=0.15

    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
    totalStride = 8
    contextAmount = 0.5
    final_sz = responseUp * (scoreSize-1) + 1
    max_translate=64

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instanceSize - self.exemplarSize) / self.total_stride + 1


def track_video(model, video):
    toc, regions = 0, []
    image_files, gt = video['image_files'], video['gt']
    ims = []
    rect=[]
    Alliou=[]


    for f, image_file in enumerate(image_files):
     
        tic = cv2.getTickCount()
        
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            update=False
            state = Siam3D_init(image_file, target_pos, target_sz,update)  # init tracker
            regions.append(gt[f])
        elif f > 0 :  # tracking
            ims.append(image_file)
            
            if f%4==0:
                state,pre = Siam3D_track(state, ims,model,regions,f)  # track
                ims = []
               
            elif f == len(image_files)-1 and len(ims)<4:
                for i in range(4-len(ims)):
                    ims.append(image_file)
                   
                state,pre = Siam3D_track(state, ims,model,regions,f)  # track

        toc += cv2.getTickCount() - tic

    
    toc /= cv2.getTickFrequency()
    args.visualization=False
    video_path = join('test', args.dataset, 'Siam3D(2D)')
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    print(result_path)
    with open(result_path, "w") as fin:
        
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))
    return f / toc





def load_dataset(dataset):
    base_path = join(realpath(dirname(__file__)), 'data', dataset)
    if not exists(base_path):
        print("Please download OTB dataset into `data` folder!")
        exit()
    json_path = join(realpath(dirname(__file__)), 'data', dataset + '.json')
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
        info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
        info[v]['name'] = v
    
    return info


def main():
    global args, v_id
    args = parser.parse_args()
   
    net = R3DNet((2, 2, 2, 2), block_type=SpatioTemporalResBlock)
    model = SiamR3D(branch=net)
    checkpoint=torch.load('model_bestPath',map_location='cuda')
    
    model.load_state_dict(checkpoint['state_dict'])                # load trained model parameters from mat file
   
    model.eval().cuda()

    dataset = load_dataset(args.dataset)
    
 
    
    fps_list = []
    for v_id, video in enumerate(dataset.keys()):
       
        fps_list.append(track_video(model, dataset[video]))
        
        # break
    print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))


if __name__ == '__main__':
    main()
