import numbers

import cv2
import numpy as np
import h5py
from PIL import Image, ImageStat, ImageOps
from collections import namedtuple
import torch
import torchvision.transforms as transforms
import math
def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    return image

# adjust learning rate according to epoch dynamically (train_Siamfc.py)
# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = np.logspace(-4, -7, num=args.numEpochs)[epoch]
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(LR,optimizer, epoch, iter):
    # assert isinstance(LR, BaseLR)
    # lr_scheduler = ListLR(
    #     LinearLR(start_lr=1e-6, end_lr=1e-1, max_epoch=5, max_iter=5000),
    #     LinearLR(start_lr=1e-1, end_lr=1e-4, max_epoch=15, max_iter=5000))

    lr = LR.get_lr(epoch, iter)
    # print(lr,iter)

    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# computes and stores the average and current value (train_Siamfc.py)
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# pad_pil and crop_pil function are used to crop image patch of specified size (pair.py)
# pad image(if context exceeds the border, pad image with average_channel / or padding)

def round_up(value):
    return round(value + 1e-6 + 1000) - 1000



def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    img_mean = [int(img[:, :, 0].mean()), int(
        img[:, :, 1].mean()), int(img[:, :, 2].mean())]
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
    # print(xmin, ymin, xmax, ymax)
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
    # cv2.imshow('1',im_patch)
    # cv2.waitKey(0)
    return im_patch, scale,xmin,ymin,xmax,ymax

def _crop_and_resize(image, center, size, out_size, pad_color):
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - image.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        image = cv2.copyMakeBorder(
            image, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=pad_color)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = image[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size))
    # plt.figure('crop_x01')
    # plt.imshow(patch)
    # plt.show()

    return patch



# if necessary, you need pad frame with avgChans/0
def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = max(0, -int(round(pos_x - c)))
    ytop_pad = max(0, -int(round(pos_y - c)))
    xright_pad = max(0, int(round(pos_x + c)) - frame_sz[1])
    ybottom_pad = max(0, int(round(pos_y + c)) - frame_sz[0])
    npad = max((xleft_pad, ytop_pad, xright_pad, ybottom_pad))
    if avg_chan is not None:
        # TODO: PIL Image doesn't allow float RGB image
        avg_chan = tuple([int(round(c)) for c in avg_chan])
        im_padded = ImageOps.expand(im, border=npad, fill=avg_chan)
    else:
        im_padded = ImageOps.expand(im, border=npad, fill=0)
    return im_padded, npad                                       # return padded frame and npad

def get_template_z2(pos_x, pos_y, z_sz, image, config):
    image = Image.open(image)                           # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean               # compute mean of three channels
    frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan) # if necessary, pad frame



    c = z_sz / 2
    tr_x = npad_z + int(round(pos_x - c))       # compute x coordinate of top-left corner
    tr_y = npad_z + int(round(pos_y - c))       # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    z_crop = frame_padded_z.crop((int(tr_x),
                              int(tr_y),
                              int(tr_x + width),
                              int(tr_y + height)))
    z_crop = z_crop.resize((config.exemplarSize, config.exemplarSize), Image.BILINEAR)
    z_crops=np.array(z_crop)

    # cv2.rectangle(z_crops, (int(127/2+target_w/2+1), int(127/2+target_h/2+1)),
    #               (int(127/2-target_w/2+1), int(127/2-target_h/2+1)),
    #               (0, 255, 0), 3)

    # cv2.rectangle(im_show, (int(anno[f+j][0]), int(anno[f+j][1])), (int(anno[f+j][2]), int(anno[f+j][3])),
    #               (0, 255, 255), 2)

    # cv2.imshow("uu",z_crops)
    # cv2.waitKey(0)
    transform = transforms.ToTensor()
    z_crop = 255.0 * transform(z_crop)
    # zcrops = torch.stack((z_crop, z_crop, z_crop))
    return z_crop

def get_template_x2(pos_x, pos_y, z_sz, image, config):
    image = Image.open(image)                           # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean               # compute mean of three channels
    frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan) # if necessary, pad frame

    c = z_sz / 2
    tr_x = npad_z + int(round(pos_x - c))       # compute x coordinate of top-left corner
    tr_y = npad_z + int(round(pos_y - c))       # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    z_crop = frame_padded_z.crop((int(tr_x),
                              int(tr_y),
                              int(tr_x + width),
                              int(tr_y + height)))
    z_crop = z_crop.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)
    z_crops=np.array(z_crop)
    # cv2.imshow("uu",z_crops)
    # cv2.waitKey(20)
    transform = transforms.ToTensor()
    z_crop = 255.0 * transform(z_crop)
    # zcrops = torch.stack((z_crop, z_crop, z_crop))
    return z_crop


# extract z_crop (Size: 127) as template
def get_template_z(target_pos, z_sz, image, exemplarSize):
    pos_x, pos_y=target_pos
    image = Image.open(image) 
                              # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean               # compute mean of three channels
    frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan) # if necessary, pad frame

    c = z_sz / 2
    tr_x = npad_z + int(round(pos_x - c))       # compute x coordinate of top-left corner
    tr_y = npad_z + int(round(pos_y - c))       # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    z_crop = frame_padded_z.crop((int(tr_x),
                              int(tr_y),
                              int(tr_x + width),
                              int(tr_y + height)))
    z_crop = z_crop.resize((exemplarSize, exemplarSize), Image.BILINEAR)
    
    # z_crops=np.array(z_crop)
    # filenamez="/home/guiyan/workspaces/liangmin/Siam3DM1/00template.jpg"
    # cv2.imwrite(filenamez, z_crops)
    
    # cv2.imshow("uu",z_crops)
    # cv2.waitKey(0)
    transform = transforms.ToTensor()
    z_crop = 255.0 * transform(z_crop)
    # zcrops = torch.stack((z_crop, z_crop, z_crop))
    return z_crop

def get_template_x1(target_pos, z_sz, image, instanceSize):
    pos_x, pos_y=target_pos
    image = Image.open(image)                           # open image
    
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    
    avg_chan = ImageStat.Stat(image).mean               # compute mean of three channels
    frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan) # if necessary, pad frame

    c = z_sz / 2
    tr_x = npad_z + int(round(pos_x - c))       # compute x coordinate of top-left corner
    tr_y = npad_z + int(round(pos_y - c))       # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    z_crop = frame_padded_z.crop((int(tr_x),
                              int(tr_y),
                              int(tr_x + width),
                              int(tr_y + height)))
    z_crop = z_crop.resize((instanceSize, instanceSize), Image.BILINEAR)
    
    # z_crops=np.array(z_crop)
    # filenamez="/home/guiyan/workspaces/liangmin/Siam3DM1/00search.jpg"
    # cv2.imwrite(filenamez, z_crops)
    
    # cv2.imshow("uu",z_crops)
    # cv2.waitKey(20)
    transform = transforms.ToTensor()
    z_crop = 255.0 * transform(z_crop)
    # zcrops = torch.stack((z_crop, z_crop, z_crop))
    return z_crop

def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):

    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = float(size_z) / s_z

    s_x = s_z * size_x / float(size_z)
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
    # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
    # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
    # cv2.imwrite('1.jpg', frame)
    return instance_img, w_x, h_x, scale_x

def _crop_and_resize(image, center, size, out_size, pad_color):
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - image.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        image = cv2.copyMakeBorder(
            image, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=pad_color)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = image[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size))
    # plt.figure('crop_x01')
    # plt.imshow(patch)
    # plt.show()

    return patch

# extract x_crop (Size: 255) as search
def get_search_x(pos_x, pos_y, scaled_search_area, image, config):
    image = Image.open(image)          # open image
    if image.mode == 'L':                               # if im is gray image, convert it to RGB
        image = image.convert('RGB')
    avg_chan = ImageStat.Stat(image).mean   # compute mean of three channels
    frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)

    # scaledInstance[2] correspondsto the maximum size of image patch
    c = scaled_search_area[2] / 2
    tr_x = npad_x + int(round(pos_x - c))               # compute x coordinate of top-left corner
    tr_y = npad_x + int(round(pos_y - c))               # compute y coordinate of top-left corner
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    search_area = frame_padded_x.crop((int(tr_x),                      # search_area corresponds to scaledInstance[2]
                                       int(tr_y),
                                       int(tr_x + width),
                                       int(tr_y + height)))
    offset_s0 = (scaled_search_area[2] - scaled_search_area[0]) / 2
    offset_s1 = (scaled_search_area[2] - scaled_search_area[1]) / 2

    crop_s0 = search_area.crop((int(offset_s0),                 # crop_x0 corresponds to scaleInstance[0]
                                int(offset_s0),
                                int(offset_s0 + scaled_search_area[0]),
                                int(offset_s0 + scaled_search_area[0])))
    crop_s0 = crop_s0.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)    # x0_crop resize to 255

    crop_s1 = search_area.crop((int(offset_s1),                 # crop_x1 corresponds to scaleInstance[1]
                                int(offset_s1),
                                int(offset_s1 + scaled_search_area[1]),
                                int(offset_s1 + scaled_search_area[1])))
    crop_s1 = crop_s1.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)    # x1_crop resize to 255

    crop_s2 = search_area.resize((config.instanceSize, config.instanceSize), Image.BILINEAR)# x2_crop resize to 255

    transfrom = transforms.ToTensor()
    crop_s0 = 255.0 * transfrom(crop_s0)
    crop_s1 = 255.0 * transfrom(crop_s1)
    crop_s2 = 255.0 * transfrom(crop_s2)
    crops = torch.stack((crop_s0, crop_s1, crop_s2))
    return crop_s1

def RandomStretch(sample, gt_w, gt_h):
    max_stretch=0.15
    scale_h = 1.0 + np.random.uniform(-max_stretch, max_stretch)
    scale_w = 1.0 + np.random.uniform(-max_stretch, max_stretch)
    # print(scale_h,scale_w)
    h, w = sample.shape[:2]
    shape = int(w * scale_w), int(h * scale_h)
    scale_w = float(w * scale_w) / w
    scale_h = float(h * scale_h) / h
    gt_w = gt_w * scale_w
    gt_h = gt_h * scale_h

    return cv2.resize(sample, shape, interpolation=cv2.INTER_LINEAR), gt_w, gt_h

#计算IOU函数
def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2. + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2. - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2. + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2. - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2. + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2. - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2. + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2. - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def box_transform(anchors, gt_box):
    eps = 1e-5

    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box


    target_x = (gt_cx - anchor_xctr) / (anchor_w + eps)
    target_y = (gt_cy - anchor_yctr) / (anchor_h + eps)
    target_w = np.log(gt_w / (anchor_w + eps))
    target_h = np.log(gt_h / (anchor_h + eps))
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target

def xywh2xyxy(rect):
    rect = np.array(rect, dtype=np.float32)
    return np.concatenate([
        rect[..., [0]], rect[..., [1]], rect[..., [2]] + rect[..., [0]] - 1,
        rect[..., [3]] + rect[..., [1]] - 1
    ],
                          axis=-1)
##中心点转为左上角

def cxy_wh_2_rect(pos, sz):
    return np.array([int(pos[0]-sz[0]/2), int(pos[1]-sz[1]/2), int(sz[0]), int(sz[1])])  # 0-index

def cxy_wh_2_rect2(rect):
    return np.array([int(rect[0]-rect[2]/2), int(rect[1]-rect[3]/2), int(rect[2]), int(rect[3])])  # 0-inde
####

def rect_2_cxy_wh2(rect):
    return np.array([int(rect[0]+rect[2]/2), int(rect[1]+rect[3]/2), int(rect[2]), int(rect[3])])  # 0-index


def  rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index



def  rect_12_rect2_wh(bbox):
    return np.concatenate([bbox[..., [0]] ,bbox[..., [1]],
                           bbox[..., [2]] - bbox[..., [0]] + 1,
                           bbox[..., [3]] - bbox[..., [1]] + 1],
                          axis=-1)

def xyxy2cxywh(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    return np.concatenate([(bbox[..., [0]] + bbox[..., [2]]) / 2,
                           (bbox[..., [1]] + bbox[..., [3]]) / 2,
                           bbox[..., [2]] - bbox[..., [0]] + 1,
                           bbox[..., [3]] - bbox[..., [1]] + 1],
                          axis=-1)


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h




def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def _postprocess_score(score, box_wh, target_sz, scale_x):

    r"""
    Perform SiameseRPN-based tracker's post-processing of score
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_sz: previous state (w & h)
    :param scale_x:
    :return:
        best_pscore_id: index of chosen candidate along axis HW
        pscore: (HW, ), penalized score
        penalty: (HW, ), penalty due to scale/ratio change
    """
    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
      
        pad = (w + h) * 0.5
       
        sz2 = (w + pad) * (h + pad)
        
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty

    penalty_k =0.21
    target_sz_in_crop = target_sz * scale_x
    
    s_c = change(
        sz(box_wh[:, 2], box_wh[:, 3]) /
        (sz_wh(target_sz_in_crop)))  # scale penalty
  
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                 (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
    penalty = np.exp(-(r_c * s_c - 1) * penalty_k)


    pscore = score*penalty
    window_influence = 0.21
    # self._state['window'] :汉宁窗
    window = np.outer(np.hanning(17), np.hanning(17))
    window = window.reshape(-1)
    pscore = pscore * (
            1 - window_influence) + window * window_influence
    best_pscore_id = np.argmax(pscore)



    return best_pscore_id, penalty

def _postprocess_box( best_pscore_id, score, box_wh, target_pos,
                     target_sz, scale_x, x_size, penalty,f):
    # print(scale_x)
    r"""
    Perform SiameseRPN-based tracker's post-processing of box
    :param score: (HW, ), score prediction
    :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
    :param target_pos: (2, ) previous position (x & y)
    :param target_sz: (2, ) previous state (w & h)
    :param scale_x: scale of cropped patch of current frame
    :param x_size: size of cropped patch
    :param penalty: scale/ratio change penalty calculated during score post-processing
    :return:
        new_target_pos: (2, ), new target position
        new_target_sz: (2, ), new target size
    """

    pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
    # about np.float32(scale_x)

    # attention!, this casting is done implicitly
    # which can influence final EAO heavily given a model & a set of hyper-parameters
    # box post-postprocessing
    
    test_lr =0.65
   
    lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
    
    
   
    #可视化（后处理之前，后处理之后）
      
    res_x =  pred_in_crop[0] +target_pos[0]- (x_size // 2)/ np.float32(scale_x)
    res_y=  pred_in_crop[1]+target_pos[1] - (x_size // 2) / np.float32(scale_x)
  
    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr


  

    new_target_pos = np.array([res_x, res_y])
    new_target_sz = np.array([res_w, res_h])
    
    return new_target_pos, new_target_sz
      
