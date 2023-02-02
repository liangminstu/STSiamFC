import glob
import pickle
import argparse, cv2, torch, json
from re import S
import numpy as np
import copy
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists
from torch.autograd import Variable

from glob import glob
from tqdm import tqdm
from PIL import Image
from toolkit.utils.region import vot_overlap, vot_float2str
import os
import sys
sys.path.append('/home/guiyan/workspaces/liangmin/Siam3DM1')
# from network.C3D import *
from network.R3DCppOp3D import *
from data_process.utils import *


from run_Siam3D import Siam3D_init,Siam3D_track,Siam3D_trackCLS

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='VOT2019', help='datasets')
# parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
#                     help='whether visualize result')


class Dataset(object):
    def __init__(self, name, dataset_root):
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = tracker_names
   

class Video(object):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        # print(root)
        # self.img_names = [os.path.join(os.path.abspath(root), os.path.join(x)) for x in img_names]
        self.img_names = [os.path.join(os.path.abspath(root), os.path.join(x)) for x in img_names]
       

        #VOT2018
        
        # self.img_names = [os.path.join(os.path.abspath(root), os.path.join(x.split('/')[0],x.split('/')[2])) for x in img_names]
        
       

        self.imgs = None

        if load_img:
            # self.imgs = [cv2.imread(x) for x in self.img_names]
            self.imgs = [x for x in self.img_names]
           
            # print(self.imgs,self.imgs.shape)
            # self.width = self.imgs[0].shape[1]
            # self.height = self.imgs[0].shape[0]
        else:
            # img = cv2.imread(self.img_names[0])
            img = self.img_names[0]
            assert img is not None, self.img_names[0]
            # self.width = img.shape[1]
            # self.height = img.shape[0]
       
            

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print(name, len(pred_traj), len(self.gt_traj), self.name)
                if store:
                    self.pred_trajs[name] = pred_traj
                else:
                    return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    def load_img(self):
        if self.imgs is None:
            # self.imgs = [cv2.imread(x) for x in self.img_names]
            self.imgs = [x for x in self.img_names]
            
            # self.width = self.imgs[0].shape[1]
            # self.height = self.imgs[0].shape[0]

    def free_img(self):
        self.imgs = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            # return cv2.imread(self.img_names[idx]), self.gt_traj[idx]
            return self.img_names[idx], self.gt_traj[idx]
        else:
            return self.imgs[idx], self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield self.imgs[i], self.gt_traj[i]
            else:
                yield cv2.imread(self.img_names[i]), self.gt_traj[i]

    def draw_box(self, roi, img, linewidth, color, name=None):
        """
            roi: rectangle or polygon
            img: numpy array img
            linewith: line width of the bbox
        """
        if len(roi) > 6 and len(roi) % 2 == 0:
            pts = np.array(roi, np.int32).reshape(-1, 1, 2)
            color = tuple(map(int, color))
            img = cv2.polylines(img, [pts], True, color, linewidth)
            pt = (pts[0, 0, 0], pts[0, 0, 1]-5)
            if name:
                img = cv2.putText(img, name, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        elif len(roi) == 4:
            if not np.isnan(roi[0]):
                roi = list(map(int, roi))
                color = tuple(map(int, color))
                img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),
                         color, linewidth)
                if name:
                    img = cv2.putText(img, name, (roi[0], roi[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        return img

    def show(self, pred_trajs={}, linewidth=2, show_name=False):
        """
            pred_trajs: dict of pred_traj, {'tracker_name': list of traj}
                        pred_traj should contain polygon or rectangle(x, y, width, height)
            linewith: line width of the bbox
        """
        assert self.imgs is not None
        video = []
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        colors = {}
        if len(pred_trajs) == 0 and len(self.pred_trajs) > 0:
            pred_trajs = self.pred_trajs
        for i, (roi, img) in enumerate(zip(self.gt_traj,
                self.imgs[self.start_frame:self.end_frame+1])):
            img = img.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = self.draw_box(roi, img, linewidth, (0, 255, 0),
                    'gt' if show_name else None)
            for name, trajs in pred_trajs.items():
                if name not in colors:
                    color = tuple(np.random.randint(0, 256, 3))
                    colors[name] = color
                else:
                    color = colors[name]
                img = self.draw_box(trajs[0][i], img, linewidth, color,
                        name if show_name else None)
            cv2.putText(img, str(i+self.start_frame), (5, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
            cv2.imshow(self.name, img)
            cv2.waitKey(40)
            video.append(img.copy())
        return video


class VOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect,
            camera_motion, illum_change, motion_change, size_change, occlusion, load_img=False):
        
        super(VOTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.tags= {'all': [1] * len(gt_rect)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion
       
        # TODO
        # if len(self.gt_traj[0]) == 4:
        #     self.gt_traj = [[x[0], x[1], x[0], x[1]+x[3]-1,
        #                     x[0]+x[2]-1, x[1]+x[3]-1, x[0]+x[2]-1, x[1]]
        #                         for x in self.gt_traj]

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        # self.tags['empty'] = np.all(1 - np.array(list(self.tags.values())),
        #         axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())
        # if not load_img:
        #     img_name = os.path.join(os.path.abspath(root), os.path.abspath(self.img_names[0]))
      
        #     img = np.array(Image.open(img_name), np.uint8)
        #     self.width = img.shape[1]
        #     self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
            if len(traj_files) == 15:
                traj_files = traj_files
            else:
                traj_files = traj_files[0:1]
            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

class VOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016', 'VOT2019'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(VOTDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        
            meta_data = json.load(f)

        

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
           
            pbar.set_postfix_str(video)
            self.videos[video] = VOTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['camera_motion'],
                                          meta_data[video]['illum_change'],
                                          meta_data[video]['motion_change'],
                                          meta_data[video]['size_change'],
                                          meta_data[video]['occlusion'],
                                          load_img=load_img)
            
        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change',
                     'size_change', 'occlusion', 'empty']




def track_video(model, video):
    toc, regions = 0, []
    frame_counter = 0
    lost_number = 0
    image_files=[]
    for idx, (img, gt_bbox) in enumerate(video):
     
        image_files.append(img)

    ims = []
    for f, image_file in enumerate(image_files):
     
        tic = cv2.getTickCount()
        
        
        
        if f == 0:  # init
            cx,cy,w,h= get_axis_aligned_bbox(np.array(gt[f]))
            target_pos=np.array([cx, cy])
            target_sz=np.array([w, h])
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            
            update=False
            state = Siam3D_init(image_file, target_pos, target_sz,update)  # init tracker
          
            regions.append(1)
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

    video_path = os.path.join('test', args.dataset, 'Siam3D19',
            'baseline', video.name)
    if not os.path.isdir(video_path):
        os.makedirs(video_path)
    # result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))

    result_path = join(video_path, '{}_001.txt'.format(video.name))
    print(result_path)
    with open(result_path, "w") as fin:
        
        for x in regions:
            if isinstance(x, int):
                fin.write("{:d}\n".format(x))
            else:
                fin.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps  Lost: {:d}'.format(
        v_id, video.name, toc, f / toc,lost_number))
    return f / toc



def main():
    global args, v_id
    args = parser.parse_args()

    net = R3DNet((2, 2, 2, 2), block_type=SpatioTemporalResBlock)
    
    # model.eval()
   
    model = SiamR3D(branch=net)
    checkpoint=torch.load('modelPath',map_location='cuda')
   
    model.load_state_dict(checkpoint['state_dict'])                # load trained model parameters from mat file
    
    model.eval().cuda()


    dataset = VOTDataset(name='VOT2019',
                        dataset_root='VOT2019',
                        load_img=True)
    
 
    
    fps_list = []
    total_lost =0
    for v_id, video in enumerate(dataset):
       
    
  
      
        fps_list.append(track_video(model, video))

    print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))


if __name__ == '__main__':
    main()
