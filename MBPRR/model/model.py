import os

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from MBPRR.net import net_common as common


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.backbone = Backbone()

        self.rootup = nn.Upsample(scale_factor=2, mode='bicubic')
        self.rfb = common.RFB_Block2(ch_in=3, ch_out=8)
        self.root = nn.Sequential(
            common.RFB_Block2(ch_in=8, ch_out=32),
            common.RFB_Block2(ch_in=32, ch_out=64)
        )

        self.upsample = common.ConvUpsampler(64, 64)

        self.conv_0_1 = common.default_conv(ch_in=64, ch_out=96, k_size=3, stride=2)
        self.conv_0_2 = common.default_conv(ch_in=96, ch_out=128, k_size=3, stride=2)
        self.conv_0_3 = common.default_conv(ch_in=128, ch_out=160, k_size=3, stride=2)
        self.conv_0_4 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_1_1 = common.default_conv(ch_in=64, ch_out=128, k_size=3, stride=1)
        self.conv_1_2 = common.default_conv(ch_in=128, ch_out=160, k_size=3, stride=2)
        self.conv_1_3 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_2_1 = common.default_conv(ch_in=256, ch_out=160, k_size=3, stride=1)
        self.conv_2_2 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_3_1 = common.default_conv(ch_in=512, ch_out=192, k_size=3, stride=1)

        self.conv_4_1 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_out_1 = common.default_conv(ch_in=192, ch_out=160, k_size=3, stride=1)
        self.conv_out_2 = common.default_conv(ch_in=160, ch_out=128, k_size=3, stride=1)

        self.cubicfilter = CubicFilter(num_in_channels=128, num_out_channels=128)

        self.inv3 = nn.Sequential(
            common.default_conv(ch_in=128, ch_out=96, k_size=1, stride=1),
            common.involution(channels=96, kernel_size=3, stride=1),
            common.ConvUpsampler(96, 96),
            common.default_conv(ch_in=96, ch_out=96, k_size=1, stride=1)
        )
        self.inv4 = nn.Sequential(
            common.default_conv(ch_in=96, ch_out=64, k_size=1, stride=1),
            common.involution(channels=64, kernel_size=3, stride=1),
            common.ConvUpsampler(64, 64),
            common.default_conv(ch_in=64, ch_out=64, k_size=1, stride=1)
        )

        self.conv_4_l = common.default_conv(ch_in=64, ch_out=32, k_size=3, stride=1, bias=True, group=False)
        self.conv_5_l = common.default_conv(ch_in=32, ch_out=16, k_size=3, stride=1, bias=True, group=False)
        self.conv_6_l = common.default_conv(ch_in=16, ch_out=1, k_size=1, stride=1, bias=True, group=False)

        self.conv_4_ab = common.default_conv(ch_in=64, ch_out=32, k_size=3, stride=1, bias=True, group=False)
        self.conv_5_ab = common.default_conv(ch_in=32, ch_out=16, k_size=3, stride=1, bias=True, group=False)
        self.conv_6_ab = common.default_conv(ch_in=16, ch_out=2, k_size=3, stride=1, bias=True, group=False)

        self.ps2 = nn.PixelShuffle(2)
        self.sf = ShuffleFeature1(3, 3)
        self.sf1 = ShuffleFeature1(64, 64)
        self.sf2 = ShuffleFeature1(8, 8)

    def forward(self, x):
        x = x.float()  # 3, 128, 128
        x = self.sf(x) + x  # 3, 128, 128

        x1_0, x1_1, x2_0, x3_0 = self.backbone(x)  # 64,128,128  128,64,64  256,32,32  512,16,16
        x1_0_up = self.upsample(x1_0)  # 64,256,256

        x0_1 = self.rootup(x)  # 3, 256, 256
        x0_1 = self.sf(x0_1) + x0_1
        x0_1 = self.rfb(x0_1)  # 8, 256, 256
        x0_1 = self.sf2(x0_1) + x0_1
        x0_1 = self.root(x0_1) + x1_0_up  # 64, 256, 256
        x0_2 = common.Mish(self.conv_0_1(x0_1))  # 96, 128, 128
        x0_3 = common.Mish(self.conv_0_2(x0_2))  # 128, 64, 64
        x0_4 = common.Mish(self.conv_0_3(x0_3))  # 160, 32, 32
        x0_5 = common.Mish(self.conv_0_4(x0_4))  # 192, 16, 16

        x1_2 = common.Mish(self.conv_1_2(x1_1)) + x0_4  # 160, 32, 32
        x1_3 = common.Mish(self.conv_1_3(x1_2)) + x0_5  # 192, 16, 16

        x2_1 = common.Mish(self.conv_2_1(x2_0)) + x1_2  # 160, 32, 32
        x2_2 = common.Mish(self.conv_2_2(x2_1)) + x1_3  # 192, 16, 16

        x3_1 = common.Mish(self.conv_3_1(x3_0)) + x2_2  # 192, 16, 16

        out_1 = torch.cat([x0_5, x1_3, x2_2, x3_1], dim=1)
        out_1 = common.Mish(self.conv_out_1(self.ps2(out_1)))  # 160, 32, 32

        out_2 = torch.cat([x0_4, x1_2, x2_1, out_1], dim=1)
        out_2_1 = common.Mish(self.conv_out_2(self.ps2(out_2))) + x0_3 + x1_1  # 128, 64, 64
        out_2 = self.cubicfilter.get_cubic_mask(out_2_1) * out_2_1  # 128, 64, 64

        x = common.Mish(self.inv3(out_2)) + x0_2  # 96, 128, 128
        x = common.Mish(self.inv4(x)) + x0_1  # 64, 256, 256

        l = common.Mish(self.conv_4_l(x))  # 32, 256, 256
        l = common.Mish(self.conv_5_l(l))  # 16, 256, 256
        l = common.Mish(self.conv_6_l(l))  # 1, 256, 256
        ab = common.Mish(self.conv_4_ab(x))  # 32, 256, 256
        ab = common.Mish(self.conv_5_ab(ab))  # 16, 256, 256
        ab = common.Mish(self.conv_6_ab(ab))  # 2, 256, 256

        return l, ab

    def model_train(self, l_lr, l_hr, ab):
        out_l_hr, out_ab = self.forward(l_lr)
        loss_1 = self.l1_loss_fun(l_hr, out_l_hr)
        loss_2 = self.l2_loss_fun(ab, out_ab)
        loss = loss_1 + loss_2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_count.update(loss.item(), l_lr.shape[0])

    def save_pkl(self, save_path, save_name):

        save_content = {
            'model': self.state_dict(),
            'epoch': self.train_epoch
        }
        if save_name == 'best':
            save_content = {
                'model': self.state_dict(),
                'epoch': self.train_epoch,
                'psnr': self.best_psnr,
                'ssim': self.best_ssim
            }

        torch.save(save_content, save_path + save_name + '-' + str(self.train_epoch).zfill(4) + '.pkl')

    def load_pkl(self, pkl_path, mode='trained', load=True):
        '''
        :param pkl_path: pkl路径
        :param mode: trained or best
        :param load: bool,是否要加载模型（训练时不用加载最佳模型）
        :return:
        '''
        success_load = False
        if not os.path.exists(pkl_path):
            print('PKL:this pkl path dose not exit!')
        else:
            pkl_list = sorted(os.listdir(pkl_path))

            if len(pkl_list) == 0:
                print('PKL:no ' + mode + ' model!')
            elif pkl_list[-1].find('.pkl') == -1:
                print('PKL:no ' + mode + ' model!')
            else:
                checkpoint = torch.load(pkl_path + pkl_list[-1])
                if load:
                    self.load_state_dict(checkpoint['model'])
                success_load = True
        if success_load:
            if mode == 'trained':
                self.train_epoch = checkpoint['epoch']
                print(mode.capitalize() + ' PKL(EPOCH:' + str(self.train_epoch - 1) + ') successfully loaded !')
            elif mode == 'best':
                self.best_epoch = checkpoint['epoch']
                self.best_psnr = checkpoint['psnr']
                self.best_ssim = checkpoint['ssim']
                print(mode.capitalize() + ' PKL(EPOCH:' + str(self.best_epoch - 1) + ') successfully loaded !')
        else:
            if mode == 'trained':
                self.train_epoch = 1
            elif mode == 'best':
                self.best_epoch = 1
                self.best_psnr = 0
                self.best_ssim = 0
        return self


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()
        self.num_out_channels = num_out_channels
        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(num_out_channels, num_out_channels * 10)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Cubic #########################
        feat = feat.float()
        batchsize = feat.shape[0]
        feat_cubic = self.upsample(feat)

        x = self.cubic_layer1(feat)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        x = np.arange(1, feat.shape[2] + 1, 1) / feat.shape[2]
        v = np.vander(x, increasing=False)
        x_axis = torch.from_numpy(v)
        y_axis = torch.from_numpy(np.transpose(v))

        a = torch.zeros((1, 10, feat.shape[2], feat.shape[3])).cuda()
        a[0][0] = x_axis ** 3
        a[0][1] = (x_axis ** 2) * y_axis
        a[0][2] = (x_axis ** 2)
        a[0][3] = torch.ones_like(x_axis) + x_axis * (y_axis ** 2)
        a[0][4] = x_axis * y_axis
        a[0][5] = x_axis
        a[0][6] = y_axis ** 3
        a[0][7] = y_axis ** 2
        a[0][8] = y_axis
        a[0][9] = torch.ones_like(x_axis)
        a = a.repeat_interleave(self.num_out_channels, 0).unsqueeze(dim=0).repeat_interleave(batchsize, 0)
        b = R.view((batchsize, self.num_out_channels, 10, 1, 1))
        cubic_mask = torch.sum(torch.mul(a, b), dim=2)
        img_cubic = feat + cubic_mask

        return img_cubic


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.smodel = list(self.model.children())[0]
        self.model1 = self.smodel[:4]
        self.model2 = self.smodel[4:9]
        self.model3 = self.smodel[9:18]
        self.model4 = self.smodel[18:27]
        self.model5 = self.smodel[27:36]

    def forward(self, x):
        x = x.float()  # 3, 128, 128
        feature1 = self.model1(x)  # 64, 128, 128
        feature2 = self.model2(feature1)  # 128, 64, 64
        feature3 = self.model3(feature2)  # 256, 32, 32
        feature4 = self.model4(feature3)  # 512, 16, 16
        return feature1, feature2, feature3, feature4


class ShuffleFeature1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ShuffleFeature1, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = common.default_conv(ch_in=1, ch_out=4, k_size=3)
        self.conv2 = common.default_conv(ch_in=4 * self.ch_in, ch_out=self.ch_out, k_size=3)
        self.se = SE_Block1(self.ch_out, reduction=1)
        self.inv = common.involution2(channels=4, kernel_size=3, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = torch.split(x, 1, dim=1)
        y = ()
        for i in x1:
            i = common.Mish(self.conv1(i))
            y1, y2 = torch.split(i, int(h / 2), dim=2)
            y1 = self.inv(y1, y2)
            y = y.__add__((y1, y2))
        for j in range(len(y)):
            if j % 2 == 0:
                if j == len(y) - 2:
                    y1 = torch.cat((y[j], y[1]), dim=2)
                else:
                    y1 = torch.cat((y[j], y[j + 3]), dim=2)
                if j == 0:
                    out = y1
                else:
                    out = torch.cat((out, y1), dim=1)
        out = common.Mish(self.conv2(out))
        out = self.se(x, out)
        return out


class SE_Block1(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _, _ = y.size()
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

