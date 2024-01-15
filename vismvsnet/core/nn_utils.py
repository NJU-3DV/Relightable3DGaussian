import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable, Any, Union
from collections import OrderedDict
import numpy as np
import itertools

cpg = 8


class GConvS2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(GConvS2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, in_channels//cpg, bias, padding_mode)
        self.groups = in_channels//cpg
    def forward(self, x):
        out = self.conv(x)
        n, c, h, w = out.size()
        if c % self.groups == 0: out = out.view(n, self.groups, c//self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)
        return out


class GConvS3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(GConvS3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, in_channels//cpg, bias, padding_mode)
        self.groups = in_channels//cpg
    def forward(self, x):
        out = self.conv(x)
        n, c, d, h, w = out.size()
        if c % self.groups == 0: out = out.view(n, self.groups, c//self.groups, d, h, w).transpose(1, 2).reshape(n, c, d, h, w)
        return out


class GConvTS2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros'):
        super(GConvTS2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, in_channels//cpg, bias, dilation, padding_mode)
        self.groups = in_channels//cpg
    def forward(self, x):
        out = self.conv(x)
        n, c, h, w = out.size()
        if c % self.groups == 0: out = out.view(n, self.groups, c//self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)
        return out


class GConvTS3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros'):
        super(GConvTS3d, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, in_channels//cpg, bias, dilation, padding_mode)
        self.groups = in_channels//cpg
    def forward(self, x):
        out = self.conv(x)
        n, c, d, h, w = out.size()
        if c % self.groups == 0: out = out.view(n, self.groups, c//self.groups, d, h, w).transpose(1, 2).reshape(n, c, d, h, w)
        return out


def multi_dims(func: Callable,
               input_: torch.Tensor,
               dim: List[int],
               keepdim: bool,
               **kwargs) -> torch.Tensor:

    num_dims = len(input_.size())
    other_dims = list(range(num_dims))
    for d in dim:
        other_dims.remove(d)
    transpose_order = dim + other_dims
    inverse = [0]*num_dims
    for i, d in enumerate(transpose_order):
        inverse[d] = i
    size = np.array(input_.size())
    input_ = input_.permute(*transpose_order).contiguous()
    input_ = input_.view(np.product(size[dim]), *size[other_dims])
    is_reduce = keepdim is not None
    keepdim = keepdim is True
    if is_reduce:
        kwargs['keepdim'] = keepdim
    input_ = func(input_, dim=0, **kwargs)
    if keepdim or not is_reduce:
        if is_reduce:
            size[dim] = 1
            input_ = input_.view(*size)
        else:
            input_ = input_.view(*size[transpose_order])
            input_ = input_.permute(*inverse).contiguous()
    return input_


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # self.bn_fn = nn.GroupNorm

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        # nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = self.bn_fn(planes)
        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = self.bn_fn(planes)
        # nn.init.constant_(self.bn2.weight, 0)
        # nn.init.constant_(self.bn2.bias, 0)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    downsample = None
    conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
    # bn_fn = nn.GroupNorm
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            bn_fn(planes * block.expansion)
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):

    def __init__(self, inplanes: int, enc: int, dec: int, initial_scale: int,
                 bottom_filters: List[int], filters: List[int], head_filters: List[int],
                 prefix: str, dim: int=2):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
        # bn_fn = nn.GroupNorm
        deconv_fn = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx==0 else 2, dim=dim)
            self.bottom_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim)
            self.enc_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2*f, f, 3, 1, 1, bias=False),
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            # nn.init.xavier_uniform_(block[0].weight)
            # nn.init.xavier_uniform_(block[1].weight)
            self.dec_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False)
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            block = nn.Sequential(*block)
            # nn.init.xavier_uniform_(block[0])
            self.head_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x, multi_scale=1):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        dec_out = [x]
        for i, b in enumerate(self.dec_blocks):
            if len(b) == 3: deconv, post_concat, res = b
            elif len(b) == 2: deconv, post_concat = b
            x = deconv(x)
            x = torch.cat([x, enc_out[-2-i]], 1)
            x = post_concat(x)
            if len(b) == 3: x = res(x)
            dec_out.append(x)
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        if multi_scale == 1: return x
        else: return dec_out[-multi_scale:]


class CSPN(nn.Module):

    def __init__(self, kernel_size, iteration, affinity_net, dim=2):
        super(CSPN, self).__init__()

        self.kernel_size = kernel_size
        self.iteration = iteration
        self.affinity_net = affinity_net
        self.dim = dim

    def gen_kernel(self, x):
        abs_sum = torch.sum(x.abs(), dim=1, keepdim=True)
        x = x / abs_sum
        sum_ = torch.sum(x, dim=1, keepdim=True)
        out = torch.cat([(1 - sum_), x], dim=1)
        out = out.contiguous()
        return out

    def im2col(self, x):
        size = x.size()
        offsets = list(itertools.product([*range(self.kernel_size//2+1), *range(-(self.kernel_size//2), 0)], repeat=self.dim))
        out = torch.cuda.FloatTensor(size[0], len(offsets), *size[2:]).zero_()
        for k, o in enumerate(offsets):
            out[[slice(size[0])] + [k] + [slice(max(0, i), min(size[2+d], size[2+d] + i)) for d, i in enumerate(o)]] = \
                x[[slice(size[0])] + [0] + [slice(max(0, -i), min(size[2+d], size[2+d] - i)) for d, i in enumerate(o)]]
        out = out.contiguous()
        return out

    def forward(self, kernel_x, x):
        out = self.affinity_net(kernel_x)
        kernel = self.gen_kernel(out)
        for _ in range(self.iteration):
            x = torch.sum(self.im2col(x) * kernel, dim=1, keepdim=True)
        return x


class CGRU(nn.Module):

    def __init__(self, in_planes, hidden_planes):
        super(CGRU, self).__init__()

        # reset_update_conv = 
        # nn.init.xavier_uniform_(reset_conv.weight)
        # reset_update_bn = 
        # nn.init.constant_(reset_bn.weight, 1)
        # nn.init.constant_(reset_bn.bias, 0)
        self.reset_update_net = nn.Sequential(
            nn.Conv2d(in_planes + hidden_planes, 2*hidden_planes, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(2*hidden_planes), 
            nn.Sigmoid()
        )

        # update_conv = nn.Conv2d(in_planes + hidden_planes, hidden_planes, 3, 1, 1, bias=False)
        # # nn.init.xavier_uniform_(update_conv.weight)
        # update_bn = nn.BatchNorm2d(hidden_planes)
        # # nn.init.constant_(update_bn.weight, 1)
        # # nn.init.constant_(update_bn.bias, 0)
        # self.update_net = nn.Sequential(update_conv, update_bn, nn.Sigmoid())

        prop_conv = nn.Conv2d(in_planes + hidden_planes, hidden_planes, 3, 1, 1, bias=False)
        # nn.init.xavier_uniform_(prop_conv.weight)
        prop_bn = nn.BatchNorm2d(hidden_planes)
        # nn.init.constant_(prop_bn.weight, 1)
        # nn.init.constant_(prop_bn.bias, 0)
        self.prop_net = nn.Sequential(prop_conv, prop_bn, nn.Tanh())

    def forward(self, x, h):
        gate_input = torch.cat([x, h], dim=1)
        ru = self.reset_update_net(gate_input)
        r = ru[:, :h.size()[1], ...]
        u = ru[:, h.size()[1]:, ...]
        concat_input = torch.cat([x, r*h], dim=1)
        update = self.prop_net(concat_input)
        output = u * h + (1 - u) * update
        return output, output      


class StackCGRU(nn.Module):

    def __init__(self, inplanes, filters, bottom_net=None, head_net=None):
        super(StackCGRU, self).__init__()

        prev_f = inplanes
        grus = []
        for f in filters:
            grus.append(CGRU(prev_f, f, f))
            prev_f = f
        self.grus = ListModule(grus)
        self.filters = filters

        self.bottom_net = bottom_net
        self.head_net = head_net

    def forward(self, xs):
        n, _, h, w = xs[0].size()
        states = [torch.cuda.FloatTensor(n, f, h, w).zero_() for f in self.filters]
        outputs = []
        for x in xs:
            out = x
            if self.bottom_net is not None:
                out = self.bottom_net(out)
            for i in range(len(self.filters)):
                out, states[i] = self.grus[i](out, states[i])
            if self.head_net is not None:
                out = self.head_net(out)
            outputs.append(out)
        return outputs


class hourglass(nn.Module):  # TODO: use batchnorm
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, inplanes*2, 3, 2, 1, bias=False),
            nn.GroupNorm(inplanes*2//cpg, inplanes*2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes*2, inplanes*2, 3, 1, 1),
            nn.GroupNorm(inplanes*2//cpg, inplanes*2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(inplanes*2, inplanes*4, 3, 2, 1, bias=False),
            nn.GroupNorm(inplanes*4//cpg, inplanes*4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(inplanes*4, inplanes*4, 3, 1, 1, bias=False),
            nn.GroupNorm(inplanes*4//cpg, inplanes*4),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv3d(inplanes*4, inplanes*4, 3, 1, 1, bias=False),
            nn.GroupNorm(inplanes*4//cpg, inplanes*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes*4, inplanes*4, 3, 1, 1, bias=False),
            nn.GroupNorm(inplanes*4//cpg, inplanes*4)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes*4, inplanes*2, 3, 2, 1, 1, bias=False),
            nn.GroupNorm(inplanes*2//cpg, inplanes*2)
        )  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes*2, inplanes, 3, 2, 1, 1, bias=False),
            nn.GroupNorm(inplanes//cpg, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16
        out = F.relu(self.res(out) + out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


def soft_argmin(volume, dim, keepdim=False, window=None):
    prob_vol = nn.Softmax(dim=dim)(volume)
    length = volume.size()[dim]
    index = torch.arange(0, length, dtype=prob_vol.dtype, device=prob_vol.device)
    index_shape = [length if i==dim else 1 for i in range(len(volume.size()))]
    index = index.reshape(index_shape)
    out = torch.sum(index * prob_vol, dim=dim, keepdim=True)
    out_sq = out.squeeze(dim) if not keepdim else out
    if window is None:
        return prob_vol, out_sq
    else:
        mask = ((index - out).abs() <= window).to(volume.dtype)
        prob_map = torch.sum(prob_vol * mask, dim=dim, keepdim=keepdim)
        return prob_vol, out_sq, prob_map


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)


def groupwise_correlation(v1, v2, groups, dim):
    # assert v1.size() == v2.size()
    size = list(v1.size())
    s1 = size[:dim]
    c = size[dim]
    s2 = size[dim+1:]
    assert c % groups == 0
    reshaped_size = s1 + [groups, c//groups] + s2
    v1_reshaped = v1.view(*reshaped_size)
    size = list(v2.size())
    s1 = size[:dim]
    c = size[dim]
    s2 = size[dim+1:]
    assert c % groups == 0
    reshaped_size = s1 + [groups, c//groups] + s2
    v2_reshaped = v2.view(*reshaped_size)
    vc = (v1_reshaped*v2_reshaped).sum(dim=dim+1)
    return vc


def bin_op_reduce(lst: List, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result
