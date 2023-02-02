import math
import pickle

import PIL.Image
import cv2
import os
from glob import glob
import random
import copy
import logging
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
sample_random = random.Random()
sample_random.seed(123456)

class Pair(data.Dataset):
    def __init__(self, root_dir, output_dir,config=None):

        self.fnames = []  # all video names in VID train/val dataset图片序列
        self.labels = []  # all annotation paths in VID train/val dataset标签序列
        
        self.root_dir = root_dir  # img dir

        self.all_data = []

        self.output_dir=output_dir
        self.video_dir=root_dir

       # # load parameters from config
        self.exemplar_size = config.exemplarSize  # template size(127)
        self.instance_size = config.instanceSize  # search size(255)
        self.scoreSize = config.scoreSize  # score Size(17)
        self.context_amount = config.context  # padding: 0.5
        self.max_stretch=0.15
        self.max_translate=64
        self.scale_resize = 0.15

    def __getitem__(self,output_dir, video_dir):
       
        vid_video_dir = os.path.join(video_dir, 'Data/VID')
        
        vid_videolist_dir=os.path.join(vid_video_dir,'train/ILSVRC2015_VID_train_0001')
        
        all_videos = glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0001/*/*'))# + \
        # glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0001/*/*')) #+ \
        # glob(os.path.join(vid_video_dir, 'val/*/*'))

            all_video=os.path.join(vid_videolist_dir,vid) 
            all_videos=glob(os.path.join(all_video,'*.JPEG'))

            trajs = {}
            for image_name in all_videos:
                
            
                img = cv2.imread(image_name)
                img_mean = tuple(map(int, img.mean(axis=(0, 1))))
                anno_name = image_name.replace('Data', 'Annotations')
                anno_name = anno_name.replace('JPEG', 'xml')
                tree = ET.parse(anno_name)
                root = tree.getroot()
                # bboxes = []
                filename = root.find('filename').text
               
                for obj in root.iter('object'):
                    
                    bbox = obj.find('bndbox')
                    bbox = list(map(int, [bbox.find('xmin').text,
                                        bbox.find('ymin').text,
                                        bbox.find('xmax').text,
                                        bbox.find('ymax').text]))
                    trkid = int(obj.find('trackid').text)
                    name=str(obj.find('name').text)
                    
                    if trkid in trajs:
                        trajs[trkid].append(filename)
                    else:
                        trajs[trkid] = [filename]
                   
                    
                    save_folder = os.path.join(output_dir,vid+'+obj'+str(trkid)+name)
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    instance_crop_size = int(
                        np.ceil((self.instance_size + self.max_translate * 2) * (1 + self.scale_resize)))
                    bbox = np.array(
                        [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                        bbox[3] - bbox[1] + 1])#中心点以及宽和高

                    instance_img, w, h, _ = self.get_instance_image(img, bbox,
                                                            self.exemplar_size, instance_crop_size,
                                                            self.context_amount,
                                                            img_mean)

                                
                    instance_img_name = os.path.join(save_folder,
                                                    ".{:0d}.x_{:.2f}_{:.2f}.jpg".format(int(trajs[trkid][-1]), w, h))
                    cv2.imwrite(instance_img_name, instance_img)
                 

        
    
  
    def round_up(self,value):
        return round(value + 1e-6 + 1000) - 1000

    def crop_and_pad(self,img, cx, cy, model_sz, original_sz, img_mean=None):

        round_up=self.round_up
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2.
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2.
        ymax = ymin + original_sz - 1

        left = int(round_up(max(0., -xmin)))
        top = int(round_up(max(0., -ymin)))
        right = int(round_up(max(0., xmax - im_w + 1)))
        bottom = int(round_up(max(0., ymax - im_h + 1)))

        xmin = int(round_up(xmin + left))
        xmax = int(round_up(xmax + left))
        ymin = int(round_up(ymin + top))
        ymax = int(round_up(ymax + top))
        r, c, k = img.shape
        if any([top, bottom, left, right]):
            # 0 is better than 1 initialization
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(
                ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(
                ymax + 1), int(xmin):int(xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):
            # zzp: use cv to get a better speed
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        scale = float(model_sz) / im_patch_original.shape[0]
        return im_patch, scale

    def get_instance_image(self,img, bbox, size_z, size_x, context_amount, img_mean=None):
        cx, cy, w, h = bbox  # float type
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
        scale_z = float(size_z) / s_z

        s_x = s_z * size_x / float(size_z)
        instance_img, scale_x = self.crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
        w_x = w * scale_x
        h_x = h * scale_x
       
        return instance_img, w_x, h_x, scale_x




class config(object):
    exemplarSize = 127
    instanceSize = 255
    scoreSize = 25
    context = 0.5
    rPos = 16
    rNeg = 0
    totalStride = 8
    ignoreLabel = -100



if __name__ == '__main__':
    config = config()
  
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    root_dir = r'/data/workspaces/datasets/ILSVRC2015/'
    
    
    outdir='/home/guiyan/workspaces/liangmin/Siam3DM1/data_process/ILSVRC2015A/train'
    get = Pair(root_dir,outdir,  config=config)
    get.__getitem__(outdir,root_dir)

