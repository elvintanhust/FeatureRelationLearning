from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import random
import math
MINIMAL = 1e-6
MAXIMAL = 1e6


class MaskConv(nn.Conv2d):
    def __init__(self, in_planes, out_planes, connect_cnt=16,
                 kernel_size=3, stride=1, groups=1, dilation=1,
                 padding=0, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MaskConv, self).__init__(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                       padding=padding, groups=groups, bias=bias, dilation=dilation,
                                       padding_mode='zeros')
        if in_planes <= 10:
            self.connect_cnt = in_planes
        else:
            self.connect_cnt = connect_cnt

        self.out_planes = out_planes
        self.in_planes = in_planes
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.mask = Parameter(torch.ones(size=(out_planes, in_planes, 1, 1), dtype=torch.float32), requires_grad=False)
        self.mask_update_enable = False

        self.out_link = (self.out_planes * self.connect_cnt) / self.in_planes

        self.correlation = torch.zeros(size=(out_planes, in_planes), dtype=torch.float32).cuda()
        self.finished_disuse = []
        self.per_disuse_cnt = max(1, math.ceil((self.in_planes - self.connect_cnt) * 0.025))

    def set_mask_update_enable(self, flag):
        self.mask_update_enable = flag
        self.correlation = self.correlation * 0.0

    def update_use_disuse(self):
        correlation = self.correlation

        index_list = list(range(correlation.size(0)))
        random.shuffle(index_list)
        for i in index_list:
            if i in self.finished_disuse:
                continue
            out_link_cnt = torch.sum(self.mask.squeeze(-1).squeeze(-1), dim=0)
            out_link = out_link_cnt - torch.mean(out_link_cnt)
            out_link = 1.0 + (torch.exp(-out_link) - torch.exp(out_link))/(torch.exp(-out_link) + torch.exp(out_link))

            corr_i = correlation[i, :]
            mask_i = self.mask[i, :, 0, 0]

            temp = torch.zeros_like(corr_i)
            temp[mask_i == 0] = MAXIMAL
            temp = temp + corr_i*out_link
            intensity, disuse_con = temp.sort(dim=-1, descending=False)

            res_cnt = int(torch.sum(self.mask[i, :, 0, 0]).detach().cpu().numpy() - self.connect_cnt)
            if res_cnt <= 0:
                self.finished_disuse.append(i)
                continue
            if res_cnt > self.per_disuse_cnt:
                self.mask.data[i, disuse_con[:self.per_disuse_cnt], 0, 0] = 0
            else:
                self.mask.data[i, disuse_con[:res_cnt], 0, 0] = 0

        disuse_ratio = torch.sum(self.mask.data).detach().cpu().numpy() / (self.out_planes * self.connect_cnt)
        return disuse_ratio

    def update_mask_conv(self, x):
        pad = self.padding[0]
        stride = self.stride[0]
        n, c, h_in, w_in = x.size()
        d, c, k, j = self.weight.size()
        x_pad = torch.zeros(n, c, h_in + 2 * pad, w_in + 2 * pad).cuda()
        if pad > 0:
            x_pad[:, :, pad:-pad, pad:-pad] = x
        else:
            x_pad = x

        x_pad = x_pad.unfold(2, k, stride)
        x_pad = x_pad.unfold(3, j, stride)
        weight = (self.weight*self.mask.expand_as(self.weight))
        out_total = torch.einsum(
            'nchwkj,dckj->ndhw',
            x_pad, weight)
        n, d, h_out, w_out = out_total.size()
        out_total = out_total.view(n, d, -1)
        act_all = torch.where(torch.relu(out_total) > 0, 1, 0)
        correlation = torch.zeros_like(self.correlation)
        for i in range(c):
            out_c = torch.einsum(
                'nhwkj,dkj->ndhw',
                x_pad[:, i, :, :, :, :], weight[:, i, :, :])
            out_c = out_c.view(n, d, -1)
            act_res = torch.where(torch.relu(out_total - out_c) > 0, 1, 0)

            Jaccard = 1.0 - torch.sum(act_res * act_all, dim=-1) / (torch.sum(act_res + act_all - act_res * act_all, dim=-1) + MINIMAL)
            intensity = torch.mean(torch.where(Jaccard > 0.05, 1.0, 0.0), dim=0)
            if torch.sum(intensity != intensity) > 0:
                correlation[:, i] = torch.ones_like(correlation[:, i]) * 0.5
            else:
                correlation[:, i] = (intensity.view(d, 1) * self.mask[:, i].view(d, 1))[:, 0]
        self.correlation[:, :] += correlation[:, :]

        out_total = out_total.view(n, d, h_out, w_out)
        if self.bias is not None:
            out_total = out_total + self.bias.view(1, -1, 1, 1)
        return out_total

    def forward(self, input):
        if not self.mask_update_enable or len(self.finished_disuse) >= self.out_planes:
            out = F.conv2d(input, self.weight*self.mask.expand_as(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = self.update_mask_conv(input)
        return out
