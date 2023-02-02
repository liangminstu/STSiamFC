

import argparse, cv2, torch, json
import numpy as np
import copy
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists
from torch.autograd import Variable

from network.R3D import *

from data_process.utils import rect_2_cxy_wh, cxy_wh_2_rect,get_template_z,\
    get_template_x1,_postprocess_score,_postprocess_box,xyxy2cxywh,overlap_ratio,xywh2xyxy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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





def Siam3D_init(image_file, target_pos, target_sz,update):
    state = dict()
    p = track_config()
    im=cv2.imread(image_file)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    wc_z = target_sz[0] + p.contextAmount * sum(target_sz)
    hc_z = target_sz[1] + p.contextAmount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_template_z(target_pos,s_z,image_file,p.exemplarSize)

    target_bb = np.zeros((4, 4))
    z_crops = torch.zeros((3, 4, 127, 127))
    if update>0:
        ts=4-update
    else:
        ts=4
    for num in range(ts):
        
        
        target_bb[num, :] = copy.copy([127/2, 127/2, target_sz[0], target_sz[1]])
        z_crops[:, num, :, :] = z_crop
    z_crops =Variable(z_crops.unsqueeze(0))
    

    
    gauss_label = gaussian_label_function(torch.from_numpy(target_bb).float(), 1/4/6, 1, 17, 127, end_pad_if_even=True, density=False,
                                        uni_bias=0)
    
                        
    state['p'] = p
    state['z_crops'] = z_crops
    state['gauss_label'] = gauss_label
    state['target_pos'] = target_pos
    state['target_sz'] = np.array(target_sz)
    return state


def Siam3D_track(state, ims,model,regions,rect,Alliou,f):
    vis=False
    p = state['p']
    gauss_label = state['gauss_label'].to(device)
    z_crops=state['z_crops'].to(device)
    target_pos = state['target_pos']
   
    pos=target_pos
    update=0
    target_sz=state['target_sz']
    
    rb=[]

    wc_z = target_sz[1] + p.contextAmount * sum(target_sz)
    hc_z = target_sz[0] + p.contextAmount * sum(target_sz)
    if len(regions)>x_crops.shape[-3]:
      
        for i in range(3):
            rb.append(regions[len(regions)-i-1])
        rb=xywh2xyxy(rb)
        x1 = rb[np.argmin(rb, axis=0)[0]][0]
        y1 = rb[np.argmin(rb, axis=0)[1]][1]
        x2 = rb[np.argmax(rb, axis=0)[2]][2]
        y2 = rb[np.argmax(rb, axis=0)[3]][3]

        w= int(x2 - x1)
        h = int(y2 - y1)
        wc_z = w + p.contextAmount * (w+h)
        hc_z = h + p.contextAmount * (w+h)

        pos[0]=int((x1+x2)/2)
        pos[1]=int((y1+y2)/2)

    
    s_z = np.sqrt(wc_z * hc_z)
    scale_z =  p.exemplarSize /s_z
  
    d_search = (p.instanceSize - p.exemplarSize) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    x_crops = torch.zeros((3, 4, 255, 255))

    for i in range(x_crops.shape[-3]):
        im=ims[i]
        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_template_x1(pos, s_x, im, p.instanceSize))
        x_crops[:,i,:,:]=x_crop
    x_crops =Variable(x_crops.unsqueeze(0)).to(device)
    
    xcorr_clss, xcorr_ctrs, xcorr_boxs = model(x_crops, z_crops,gauss_label)
    
    fcos_cls_prob_final = torch.sigmoid(xcorr_clss)
    fcos_ctr_prob_final = torch.sigmoid(xcorr_ctrs)
    scores = fcos_cls_prob_final * fcos_ctr_prob_final
    i=0

    for j in range(x_crops.shape[-3]):
            if j>0:
                if ims[j]==ims[j-1]:
                  
                    break

            box = xcorr_boxs[i][j].detach().cpu().numpy()  # 回归后处理得到的box
            ctr = fcos_ctr_prob_final[i][j].detach().cpu().numpy() 
            score = (scores[i][j].detach().cpu().numpy())[:, 0]  # cls*ctr
            box_wh = xyxy2cxywh(box)  # 将左上角右下角坐标转换成中心点和W\H
            top =np.max(ctr)
          

            # score post-processing--target_sz:目标框的w和H
            best_pscore_id,  penalty = _postprocess_score(
                score, box_wh, target_sz, scale_z)
          
            new_target_pos, new_target_sz = _postprocess_box(
                best_pscore_id, score, box_wh, pos, target_sz, scale_z,
                p.instanceSize, penalty,f)

            new_target_pos[0] = round(max(0, min(state['im_w'], new_target_pos[0])),1)
            new_target_pos[1] = round(max(0, min(state['im_h'], new_target_pos[1])),1)
            new_target_sz[0] = round(max(10, min(state['im_w'], new_target_sz[0])),1)
            new_target_sz[1] = round(max(10, min(state['im_h'], new_target_sz[1])),1)
            pre = (new_target_pos[0], new_target_pos[1], new_target_sz[0],new_target_sz[1])
            

            prebox=([((pre[0] - pre[2]/2)), ((pre[1] - pre[3]/2)),((pre[2])),((pre[3]))])
            prebox1=( [('%.4f'%prebox[0]),('%.4f'%prebox[1]),('%.4f'%prebox[2]),('%.4f'%prebox[3])])
            prelwh = ([int(pre[0] - pre[2]/2), int(pre[1] - pre[3]/2), int(pre[0]+pre[2]/2),int(pre[1]+pre[3]/2)])
            
            prebox = np.array(prebox)
           
            pre = np.array(pre).astype(np.float)
            prelwh=np.array(prelwh).astype(np.float)
            
            Alliou.append(overlap_ratio(np.array(rect[j]),prebox))
            regions.append(prebox)

            if 0.8>top>0.3:
                update+=1

                state = Siam3D_init(ims[j], state['target_pos'], state['target_sz'], update)

    
    state['target_pos'] = new_target_pos
    state['target_sz'] = prebox[2:]
    state['regions'] = regions
   
    return state,prebox,Alliou




    
   
    return state,prebox,Alliou
