import math
import os

import torch
import scipy
import torch.nn as nn

from sklearn import preprocessing
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
import numpy as np


from data_process.gauss import gaussian_label_function
from network.featureStacknorm import FeatureDescend


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, with_relu=True):
        super(SpatioTemporalConv, self).__init__()

        self.with_relu = with_relu

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, bias=bias)
       
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        if self.with_relu:
            x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=[1, 2, 2])
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=[1, 2, 2])
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

class conv2d_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.

        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size   pad: int
            padding on
            、each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv2d_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class conv3d_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.

        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size   pad: int
            padding on
            、each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv3d_bn_relu, self).__init__()
        self.conv = nn.Conv3d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm3d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

       
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 1, 1], padding=[1, 4, 4])
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 128, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(128, 128, 3, layer_sizes[3], block_type=block_type, downsample=True)
      

    def forward(self, x):
        
        x1 = self.conv1(x)
       
        x2 = self.conv2(x1)
       
        x3 = self.conv3(x2)
        
        x4 = self.conv4(x3)
        
        x5 = self.conv5(x4)
       
        return x5


class SiamR3D(nn.Module):
    def __init__(self, branch):
        super(SiamR3D, self).__init__()
        self.branch = branch
        # self.bn_adjust = nn.Conv3d(1, 1, 1, 1,1)
        self.bn_adjust = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self._initialize_weights()
        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))
        self.total_stride = 8
        self.score_size = 17
        self.score_offset = (255 - 1 - (self.score_size - 1) * self.total_stride) // 2

        head_width = 128

        # feature adjustment
        self.FEAT=FeatureDescend()
        self.r_z_k = conv3d_bn_relu(head_width,
                                  head_width,
                                  1,
                                  (1,3,3),
                                  0,
                                  has_relu=False)
        self.c_z_k = conv3d_bn_relu(head_width,
                                  head_width,
                                  1,
                                  (1,3,3),
                                  0,
                                  has_relu=False)
        self.r_x = conv3d_bn_relu(head_width, head_width, 1, (1,3,3), 0, has_relu=False)
        self.c_x = conv3d_bn_relu(head_width, head_width, 1, (1,3,3), 0, has_relu=False)

        self.cls_score_p5 = conv2d_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.ctr_score_p5 = conv2d_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.bbox_offsets_p5 = conv2d_bn_relu(head_width,
                                            4,
                                            stride=1,
                                            kszie=1,
                                            pad=0,
                                            has_relu=False)

        self.cls_conv3x3 = conv2d_bn_relu(head_width,
                                   head_width,
                                   stride=1,
                                   kszie=1,
                                   pad=0,
                                   has_bn=True)

        self.bbox_conv3x3 = conv2d_bn_relu(head_width,
                                    head_width,
                                    stride=1,
                                    kszie=1,
                                    pad=0,
                                    has_bn=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def xcorr_depthwise(self, x, kernel):
        r"""
        Depthwise cross correlation. e.g. used for template matching in Siamese tracking network

        Arguments
        ---------
        x: torch.Tensor
            feature_x (e.g. search region feature in SOT)
        kernel: torch.Tensor
            feature_z (e.g. template feature in SOT)

        Returns
        -------
        torch.Tensor
            cross-correlation result
        """
        batch = int(kernel.size(0))
        channel = int(kernel.size(1))
        x = x.view(1, int(batch * channel), int(x.size(2)), int(x.size(3)))
        kernel = kernel.view(batch * channel, 1, int(kernel.size(2)),
                             int(kernel.size(3)))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, int(out.size(2)), int(out.size(3)))
        return out

    def get_xy_ctr_np(self, score_size, score_offset, total_stride):
        """ generate coordinates on image plane for score map pixels (in numpy)
        """
        batch, fm_height, fm_width = 1, score_size, score_size

        y_list = np.linspace(0., fm_height - 1.,
                             fm_height).reshape(1, fm_height, 1, 1)
        y_list = y_list.repeat(fm_width, axis=2)
        x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
        x_list = x_list.repeat(fm_height, axis=1)
        xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride

        xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
            batch, -1,
            2)  # .broadcast([batch, fm_height, fm_width,
        xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))


        return xy_ctr

    def get_box(self, xy_ctr, offsets):

        offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
        offsets = offsets.reshape(offsets.shape[0], -1, 4)
        xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
        xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])


        bboxes_pred = torch.cat([xy0, xy1], 2)

        return bboxes_pred

    def forward(self, x, z,gauss_label):  # x denote search, z denote template
        # backbone feature
        x = self.branch(x)  # search feature
        z = self.branch(z)  # template feature
        xcorr_cls = []
        xcorr_ctr = []
        xcorr_box = []

        #分类分支
        c_z_k = self.c_z_k(z)
        c_x = self.c_x(x)
       
        #回归分支
        r_z_k = self.r_z_k(z)#(b,c,n,H,W)
        r_x = self.r_x(x)

        # feature Stack(search feature)
        c_x = self.FEAT(c_x, c_z_k, gauss_label)
        r_x = self.FEAT(r_x, r_z_k, gauss_label)


        # template meaning
        c_z = torch.mean(c_z_k.reshape(c_z_k.shape[0], -1, c_z_k.shape[-3], c_z_k.shape[-2], c_z_k.shape[-1]), dim=2)
        r_z = torch.mean(r_z_k.reshape(r_z_k.shape[0], -1, r_z_k.shape[-3], r_z_k.shape[-2], r_z_k.shape[-1]), dim=2)
        
        for i in range(x.size(2)):


            # feature matching
            c_out = self.xcorr_depthwise(c_x[:,:,i,:,:], c_z)
            r_out = self.xcorr_depthwise(r_x[:,:,i,:,:], r_z)


            # head
            # classification score
            cls_score = self.cls_score_p5(c_out)  # todo
            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)

            # center-ness score
            ctr_score = self.ctr_score_p5(c_out)  # todo
            ctr_score = ctr_score.permute(0, 2, 3, 1)

            ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)

            # regression

            offsets = self.bbox_offsets_p5(r_out)

            offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride


            self.fm_ctr = self.get_xy_ctr_np(self.score_size, self.score_offset,
                                             self.total_stride)
            self.fm_ctr.require_grad = False
            fm_ctr = self.fm_ctr.to(offsets.device)


            bbox = self.get_box(fm_ctr, offsets)


            xcorr_cls.append(cls_score.unsqueeze(1))
            xcorr_ctr.append(ctr_score.unsqueeze(1))
            xcorr_box.append(bbox.unsqueeze(1))


        return  torch.cat(xcorr_cls, dim=1), torch.cat(xcorr_ctr, dim=1), torch.cat(xcorr_box,dim=1)

    def load_params(self, net_path):
        checkpoint = torch.load(net_path)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)

    def load_params_from_mat(self, net_path):
        params_names_list, params_values_list = load_matconvnet(net_path)
        params_values_list = [torch.from_numpy(p) for p in params_values_list]  # values convert numpy to Tensor

        for index, param in enumerate(params_values_list):
            param_name = params_names_list[index]
            if 'conv' in param_name and param_name[-1] == 'f':
                param = param.permute(3, 2, 0, 1)
            param = torch.squeeze(param)
            params_values_list[index] = param

        net = nn.Sequential(
            self.branch.conv1,
            self.branch.conv2,
            self.branch.conv3,
            self.branch.conv4,
            self.branch.conv5
        )

        for index, layer in enumerate(net):
            layer[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % (index + 1))]
            layer[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % (index + 1))]

            if index < len(net) - 1:
                layer[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % (index + 1))]
                layer[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % (index + 1))]
                bn_moments = params_values_list[params_names_list.index('bn%dx' % (index + 1))]
                layer[1].running_mean[:] = bn_moments[:, 0]
                layer[1].running_var[:] = bn_moments[:, 1] ** 2
            else:
                self.bn_adjust.weight.data[:] = params_values_list[params_names_list.index('adjust_f')]
                self.bn_adjust.bias.data[:] = params_values_list[params_names_list.index('adjust_b')]


def load_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')  # get net
    params = net_dot_mat['params']  # get net/params
    params = params[0][0]
    params_names = params['name'][0]  # get net/params/name
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]  # get net/params/val
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    template = torch.randn(1, 3, 3, 127, 127)
    search = torch.randn(1, 3, 3, 255, 255)
    # inputs = torch.rand(1, 3, 3, 112, 112)

    target_bb = torch.tensor([[10,20,50,50],[30,50,100,100],[100,100,120,120]])

    # output_sigma = settings.output_sigma_factor / settings.search_area_factor
    output_sigma = 1 / 4 / 6
    gauss_label = gaussian_label_function(target_bb, output_sigma, 1, 15, 127, end_pad_if_even=True, density=False,
                                          uni_bias=0)

    net = R3DNet((2, 2, 2, 2), block_type=SpatioTemporalResBlock)
    xcor = SiamR3D(branch=net)


    print(xcor(search, template,gauss_label)[2].shape)


