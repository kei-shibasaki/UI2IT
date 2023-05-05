import torch
from torch import nn
import torch.nn.functional as F

import itertools

### https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            LayerNorm2d(dim), 
            nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True), 
            nn.Conv2d(2*dim, 2*dim, kernel_size=3, padding=1, groups=2*dim), 
            nn.GELU(), 
            nn.Conv2d(2*dim, dim, kernel_size=1, bias=True)
        )
        self.conv_block2 = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(2*dim, dim, kernel_size=1, bias=True)
        )
    def forward(self, x):
        x = self.conv_block1(x) + x
        x = self.conv_block2(x) + x 
        return x



class LaplacianPyramid(nn.Module):
    def __init__(self, num_high=3):
        super(LaplacianPyramid, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image
    
"""
class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(2*dim, 2*dim, kernel_size=3, padding=1, groups=2*dim)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(2*dim, dim, kernel_size=1, bias=True)
    
    def forward(self, x):
        a = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        return a + x

class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.alpha = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True)
        self.ptf = nn.GELU()
        self.conv2 = nn.Conv2d(2*dim, dim, kernel_size=1, bias=True)
    
    def forward(self, x):
        a = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.ptf(x)
        x = self.conv2(x)
        return a + x
"""