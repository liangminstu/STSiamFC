import cv2 as cv
import os
import glob
import random
import copy
import torch.utils.data as data


from data_process.createlabel import make_densebox_target
from data_process.utils import *
from data_process.gauss import gaussian_label_function



sample_random = random.Random()
sample_random.seed(123456)

class Pair(data.Dataset):

    def __init__(self, root_dir, subset='train', transform = None, config=None, pairs_per_video=25,
                 frame_range=100, rand_choice=True, process=False):
        # self.seq_dirs = sorted(glob.glob(os.path.join(root_dir, 'DAVIS/Annotations/480p/*')))

        self.fnames = []  # all video names in VID train/val dataset图片序列
        self.labels = []  # all annotation paths in VID train/val dataset标签序列
        self.anno_dirs = []  # bbox信息
        self.subset = subset  # train orr vaild
        self.pairs_per_video = pairs_per_video  # 视频对
        self.rand_choice = rand_choice  # random
        self.frame_range = frame_range  # 帧范围
        self.root_dir = root_dir  # img dir
        self.process = process  # pre-process
        self.all_data = []
        self.transform=transform
       # load parameters from config
        self.exemplarSize = config.exemplarSize  # template size(127)
        self.random_crop_size = config.instanceSize  # search size(255)
        self.scoreSize = config.scoreSize  # score Size(25)
        self.context = config.context  # padding: 0.5
        self.rPos = config.rPos  # 16
        self.rNeg = config.rNeg  # 0
        self.totalStride = config.totalStride  # 8
        self.ignoreLabel = config.ignoreLabel  # -100
        self.max_stretch=0.15
        self.max_translate=64


        if subset == 'train':

            self.seq_dirs = sorted(glob.glob(os.path.join(root_dir, 'train/*')))
            label_path=os.path.join(root_dir, 'train')
            for label in sorted(os.listdir(label_path)):

                self.labels.append(label)
                for fname in os.listdir(os.path.join(label_path, label)):
                    self.fnames.append(os.path.join(label_path, label, fname))
                    self.all_data.append(self.fnames)

        if subset == 'val':

            self.seq_dirs = sorted(glob.glob(os.path.join(root_dir, 'val/*')))
            label_path=os.path.join(root_dir, 'val')

            for label in sorted(os.listdir(label_path)):
                
                self.labels.append(label)
                for fname in os.listdir(os.path.join(label_path, label)):
                    self.fnames.append(os.path.join(label_path, label, fname))
                    self.all_data.append(self.fnames)
   

        #产生抽取图片对的随机数
        self.num = len(self.labels)
        # print(self.num,self.labels)
        self.indices = np.arange(0, self.num, dtype=int)
        self.indices = np.tile(self.indices,
                               self.pairs_per_video)  # the len of self.indices denote the number of image pairs

    def __getitem__(self, index):
    # global logger
    # logger = logging.getLogger('global')  # 加载全局日志
    # if rand_choice, select a video in VID randomly

        if self.rand_choice:
            index = np.random.choice(self.indices)


        # choose a object(track id) randomly
        #对图片进行归一化
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
        )

        # 根据随机产生的Index选择对应的类别文件

        img_files = []
        for f in os.listdir(self.seq_dirs[index]):
            img_files.append(os.path.join(self.seq_dirs[index], f))
        # print(self.seq_dirs[index])

        while True:
            # print(len(img_files))
            if len(img_files)<15:
                index = np.random.choice(self.indices)
                img_files = []
                for f in os.listdir(self.seq_dirs[index]):
                    img_files.append(os.path.join(self.seq_dirs[index], f))
            else:
                
                break
       

        # 随机选中模板和搜索图片帧并读取
        rand_z, rand_x = self.sample_pair(len(img_files)-3)   # select z(template) and x(search) randomly



        #产生3组数据：初始化能够装的下15张图片的信息

        crop_zs = np.zeros((3,3, 127, 127))
        crop_xs = np.zeros((3, 3, 255, 255))

        cls_labels = np.zeros((3,17*17,1))
        ctr_labels = np.zeros((3,17*17,1))
        box_labels = np.zeros((3, 17*17, 4))
    
        target_bb = np.zeros((3,4))

        rin = 0

        #产生搜索图片（中间图片）x
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        stretch=np.random.randint(- self.max_translate, self.max_translate + 1)
        for r in range(rand_x,rand_x+3):

            # 产生模板图片
            img_z = cv2.imread(img_files[rand_z])  # open and read z(template)
            gw, gh = float(img_files[rand_z].split(
                '_')[-2]), float(img_files[rand_z].split('_')[-1][:-4])


            crop_z, _,a,c,d,f = crop_and_pad(img_z, (img_z.shape[1] - 1) / 2, (img_z.shape[0] - 1) / 2,
                                                self.exemplarSize, self.exemplarSize)
            imz_h, imz_w ,_= crop_z.shape
            cy_t = (imz_h - 1.) / 2  # 裁剪图片的中心点
            cx_t = (imz_w - 1.) / 2
            target_bb[rin, :] = copy.copy([cx_t, cy_t, gw, gh])
          
           
            rand_z+=2

            #产生搜索区域图片
            imgx = cv2.imread(img_files[r])  # open and read z(template)
            gtw, gth = float(img_files[r].split(
                '_')[-2]), float(img_files[r].split('_')[-1][:-4])


            #将预处理得到的图片进行处理

            img_x, gt_w, gt_h = self.RandomStretch(imgx, gtw, gth,scale_h, scale_w)
            # img_x, gt_w, gt_h = self.RandomStretch(imgx, gtw, gth)

            # 获取搜索图片在原图中的真实位置
            im_h, im_w, _ = img_x.shape
            cy_o = (im_h - 1.) / 2  # 裁剪图片的中心点
            cx_o = (im_w - 1.) / 2

            cy=cy_o+stretch
            cx=cx_o+stretch

            crop_x, scale,xmin,ymin,xmax,ymax = crop_and_pad(
                img_x, cx, cy, self.random_crop_size, self.random_crop_size)

            gt_cx = int(cx_o - xmin)
            gt_cy = int(cy_o - ymin)
            gt_w = int(gt_w)
            gt_h = int(gt_h)

    


            crop_zs[:, rin, :, :] = copy.copy(transform1(crop_z))
            crop_xs[:, rin, :, :] = copy.copy(transform1(crop_x))

         
            cls_label, ctr_label,box_label = make_densebox_target(np.asarray([[gt_cx-gt_w/2, gt_cy-gt_h/2, gt_cx+gt_w/2, gt_cy+gt_h/2]]))


            cls_labels[rin,:,:]=copy.copy(cls_label)
            ctr_labels[rin, :,:] = copy.copy(ctr_label)
            box_labels[rin,:,:]=copy.copy(box_label)
            

            rin+=1

        cls_labels = torch.from_numpy(cls_labels).float()  # convert numpy to Tensor
        ctr_labels = torch.from_numpy(ctr_labels).float()
        box_labels = torch.from_numpy(box_labels).float()
        target_bb = torch.from_numpy(target_bb).float()
        gauss_label = gaussian_label_function(target_bb, 1/4/6, 1, 15, 127, end_pad_if_even=True, density=False,
                                            uni_bias=0)


        return crop_zs, crop_xs ,cls_labels,ctr_labels,box_labels,gauss_label

    def round_up(self,value):
        return round(value + 1e-6 + 1000) - 1000

    def crop_and_pad(self,img, cx, cy, model_sz, original_sz, img_mean=None):
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2.
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2.
        ymax = ymin + original_sz - 1

        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))
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

    def RandomStretch(self, sample, gt_w, gt_h,scale_h,scale_w):
        # scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        # scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)

        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = float(w * scale_w) / w
        scale_h = float(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, interpolation=cv2.INTER_LINEAR), gt_w, gt_h

    def __len__(self):
        return len(self.indices)

    def sample_pair(self, n):
        rand_z = np.random.randint(n-10)  # select a image randomly as z(template)
        self.frame_range=n
        if self.frame_range == 0:
            return rand_z, rand_z
        possible_x = np.arange(rand_z - self.frame_range,
                               rand_z + self.frame_range)  # get possible search(x) according to frame_range

        possible_x = np.intersect1d(possible_x, np.arange(n))  # remove impossible x(search)返回两个元素组共同的元素

        possible_x = possible_x[possible_x != rand_z]  # z(template) and x(search) cannot be same

        rand_x = np.random.choice(possible_x)  # select x from possible_x randomly
        return rand_z, rand_x


class config(object):
    total_stride= 8
    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
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
    root_dir = r'/home/guiyan/workspaces/liangmin/Siam3DM1/data_process/ILSVRC2015/'
    get = Pair(root_dir,subset='val', transform=transforms_train, config=config)
    get.__getitem__(20)
