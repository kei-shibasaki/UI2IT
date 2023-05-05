import torch
from torch import nn
from models.layers import ConvBlock, LayerNorm2d

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        in_channels = opt.in_channels
        out_channels = opt.out_channels

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_block_{i}_{j}', ConvBlock(chan))
                
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_block_{i}', ConvBlock(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_block_{i}_{j}', ConvBlock(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    def forward(self, x, mask):
        # x: (B, C, H, W)
        inp = torch.cat([x, mask], dim=1)
        x = self.intro(inp)

        encs = []
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                x = getattr(self, f'enc_block_{i}_{j}')(x)
            encs.append(x)
            x = getattr(self, f'enc_down_{i}')(x)

        for i in range(self.middle_blk_num):
            x = getattr(self, f'middle_block_{i}')(x)

        for i, (num, enc) in enumerate(zip(self.dec_blk_nums, encs[::-1])):
            x = getattr(self, f'dec_upconv_{i}')(x)
            x = getattr(self, f'dec_up_{i}')(x)
            x = x + enc
            for j in range(num):
                x = getattr(self, f'dec_block_{i}_{j}')(x)

        x = self.ending(x) + self.expand_dims(inp)

        out_mask = x[:,0,:,:].unsqueeze(0)
        foreground = x[:,1:4,:,:]
        background = x[:,4:7,:,:]
        output = out_mask*foreground + (1-out_mask)*background
        return out_mask, foreground, background, output


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        in_channels = opt.in_channels
        out_channels = opt.out_channels

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_block_{i}_{j}', ConvBlock(chan))
                
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_block_{i}', ConvBlock(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_block_{i}_{j}', ConvBlock(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    def forward(self, x, mask):
        # x: (B, C, H, W)
        inp = torch.cat([x, mask], dim=1)
        x = self.intro(inp)

        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                x = getattr(self, f'enc_block_{i}_{j}')(x)
            x = getattr(self, f'enc_down_{i}')(x)

        for i in range(self.middle_blk_num):
            x = getattr(self, f'middle_block_{i}')(x)

        for i, num in enumerate(self.dec_blk_nums):
            x = getattr(self, f'dec_upconv_{i}')(x)
            x = getattr(self, f'dec_up_{i}')(x)
            for j in range(num):
                x = getattr(self, f'dec_block_{i}_{j}')(x)

        x = self.ending(x)

        out_mask = x[:,0,:,:].unsqueeze(0)
        foreground = x[:,1:4,:,:]
        background = x[:,4:7,:,:]
        output = out_mask*foreground + (1-out_mask)*background
        return out_mask, foreground, background, output


class UNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.enc_blk_nums = opt.enc_blk_nums
        self.middle_blk_num = opt.middle_blk_num
        self.dec_blk_nums = opt.dec_blk_nums

        in_channels = opt.in_channels
        out_channels = opt.out_channels

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=opt.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        chan = opt.width
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                setattr(self, f'enc_block_{i}_{j}', self.conv_block(chan))
            setattr(self, f'enc_down_{i}', nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        for i in range(self.middle_blk_num):
            setattr(self, f'middle_block_{i}', self.conv_block(chan))

        for i, num in enumerate(self.dec_blk_nums):
            setattr(self, f'dec_upconv_{i}', nn.Conv2d(chan, chan * 2, 1, bias=False))
            setattr(self, f'dec_up_{i}', nn.PixelShuffle(2))
            chan = chan // 2
            for j in range(num):
                setattr(self, f'dec_block_{i}_{j}', self.conv_block(chan))
        
        self.ending = nn.Conv2d(in_channels=opt.width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
    
    def conv_block(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), 
            LayerNorm2d(dim), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        # x: (B, C, H, W)
        inp = torch.cat([x, mask], dim=1)
        x = self.intro(inp)

        encs = []
        for i, num in enumerate(self.enc_blk_nums):
            for j in range(num):
                x = getattr(self, f'enc_block_{i}_{j}')(x)
            encs.append(x)
            x = getattr(self, f'enc_down_{i}')(x)

        for i in range(self.middle_blk_num):
            x = getattr(self, f'middle_block_{i}')(x)

        for i, (num, enc) in enumerate(zip(self.dec_blk_nums, encs[::-1])):
            x = getattr(self, f'dec_upconv_{i}')(x)
            x = getattr(self, f'dec_up_{i}')(x)
            x = x + enc
            for j in range(num):
                x = getattr(self, f'dec_block_{i}_{j}')(x)

        x = self.ending(x) + self.expand_dims(inp)
        return torch.sigmoid(x)