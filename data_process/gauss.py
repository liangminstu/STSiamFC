import cv2
import numpy
import torch
import math


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True, density=False,
                            uni_bias=0):
    """Construct Gaussian label function."""

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)#.to(target_bb.device)
    feat_sz = torch.Tensor(feat_sz)#.to(target_bb.device)
    

    target_center = (target_bb[:, 0:2] )

    target_center_norm = (target_center - image_sz / 2) / image_sz



    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)
   
    
    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label

if __name__ == '__main__':
    target_bb= torch.tensor([[10,20,50,50],[30,50,100,100],[100,100,120,120]])

    # output_sigma = settings.output_sigma_factor / settings.search_area_factor
    output_sigma = 1/4 / 6
    gauss_label =gaussian_label_function(target_bb, output_sigma,1, 15,127, end_pad_if_even=True, density=False,
                            uni_bias=0)
    
    
