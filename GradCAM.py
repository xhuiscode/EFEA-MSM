import os
import numpy as np
from typing import List, Callable
from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
from torch import Tensor
from models.modulator import Modulator


def main():
    # 这个下面放置你网络的代码，因为载入权重的时候需要读取网络代码，这里我建议直接从自己的训练代码中原封不动的复制过来即可，我这里因为跑代码使用的是Resnet，所以这里将resent的网络复制到这里即可

    class eca_block(nn.Module):
        def __init__(self, channel, kernel_size=3):
            super(eca_block, self).__init__()
            # kernel_size = int(abs(math.log(channel, 2) + b) / gama)
            # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            b, c, h, w = x.size()
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            return x * y.expand_as(x)

    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    class LocalFeatureExtractor(nn.Module):

        def __init__(self, inplanes, planes, index):
            super(LocalFeatureExtractor, self).__init__()
            self.index = index

            norm_layer = nn.BatchNorm2d
            self.relu = nn.ReLU()

            self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.bn1_1 = norm_layer(planes)
            self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.bn1_2 = norm_layer(planes)

            self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.bn2_1 = norm_layer(planes)
            self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.bn2_2 = norm_layer(planes)

            self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.bn3_1 = norm_layer(planes)
            self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.bn3_2 = norm_layer(planes)

            self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
            self.bn4_1 = norm_layer(planes)
            self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.bn4_2 = norm_layer(planes)

        def forward(self, x):
            patch_11 = x[:, :, 0:28, 0:28]
            patch_21 = x[:, :, 28:56, 0:28]
            patch_12 = x[:, :, 0:28, 28:56]
            patch_22 = x[:, :, 28:56, 28:56]

            out_1 = self.conv1_1(patch_11)
            out_1 = self.bn1_1(out_1)
            out_1 = self.relu(out_1)
            out_1 = self.conv1_2(out_1)
            out_1 = self.bn1_2(out_1)
            out_1 = self.relu(out_1)

            out_2 = self.conv2_1(patch_21)
            out_2 = self.bn2_1(out_2)
            out_2 = self.relu(out_2)
            out_2 = self.conv2_2(out_2)
            out_2 = self.bn2_2(out_2)
            out_2 = self.relu(out_2)

            out_3 = self.conv3_1(patch_12)
            out_3 = self.bn3_1(out_3)
            out_3 = self.relu(out_3)
            out_3 = self.conv3_2(out_3)
            out_3 = self.bn3_2(out_3)
            out_3 = self.relu(out_3)

            out_4 = self.conv4_1(patch_22)
            out_4 = self.bn4_1(out_4)
            out_4 = self.relu(out_4)
            out_4 = self.conv4_2(out_4)
            out_4 = self.bn4_2(out_4)
            out_4 = self.relu(out_4)

            out1 = torch.cat([out_1, out_2], dim=2)
            out2 = torch.cat([out_3, out_4], dim=2)
            out = torch.cat([out1, out2], dim=3)

            return out

    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride):
            super(InvertedResidual, self).__init__()

            if not (1 <= stride <= 3):
                raise ValueError('illegal stride value')
            self.stride = stride

            branch_features = oup // 2
            assert (self.stride != 1) or (inp == branch_features << 1)

            if self.stride > 1:
                self.branch1 = nn.Sequential(
                    depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_features),
                    nn.ReLU(inplace=True))

            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                )

        def forward(self, x):
            if self.stride == 1:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)
            else:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

            out = channel_shuffle(out, 2)

            return out

    class EfficientFace(nn.Module):

        def __init__(self, stages_repeats, stages_out_channels, num_classes=8):
            super(EfficientFace, self).__init__()

            if len(stages_repeats) != 3:
                raise ValueError('expected stages_repeats as list of 3 positive ints')
            if len(stages_out_channels) != 5:
                raise ValueError('expected stages_out_channels as list of 5 positive ints')
            self._stage_out_channels = stages_out_channels

            input_channels = 3
            output_channels = self._stage_out_channels[0]
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                       nn.BatchNorm2d(output_channels),
                                       nn.ReLU(inplace=True), )
            input_channels = output_channels

            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
            for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
                seq = [InvertedResidual(input_channels, output_channels, 2)]
                for i in range(repeats - 1):
                    seq.append(InvertedResidual(output_channels, output_channels, 1))
                setattr(self, name, nn.Sequential(*seq))
                input_channels = output_channels

            self.local = LocalFeatureExtractor(29, 116, 1)
            self.modulator = Modulator(116)

            output_channels = self._stage_out_channels[-1]

            self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(output_channels),
                                       nn.ReLU(inplace=True), )

            self.fc = nn.Linear(output_channels, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.modulator(self.stage2(x)) + self.local(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = x.mean([2, 3])
            x = self.fc(x)

            return x

    def efficient_face():
        model = EfficientFace([4, 8, 4], [29, 116, 232, 464, 1024])
        return model

    net = efficient_face()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(
        torch.load("./best_EfficienFace.pth", map_location=device), strict=False)  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可

    target_layers = [net.stage4]  # 这里是 看你是想看那一层的输出，我这里是打印的resnet最后一层的输出，你也可以根据需要修改成自己的
    print(target_layers)
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932], std=[0.20735591, 0.18981615, 0.18132027])
    ])
    # 导入图片
    # img_path = "./test1.jpg"  # 这里是导入你需要测试图片
    # img_path = r"D:\edgexiazai\EfficientFace1\RAF-DB\test\0\test_1427_aligned.jpg"
    # image_size = 128  # 训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')  # 将图片转成RGB格式的
    # img = np.array(img, dtype=np.uint8)  # 转成np格式
    # img = center_crop_img(img, image_size)  # 将测试图像裁剪成跟训练图片尺寸相同大小的
    #
    # # [C, H, W]
    # img_tensor = data_transform(img)  # 简单预处理将图片转化为张量
    # # expand batch dimension
    # # [C, H, W] -> [N, C, H, W]
    # input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 增加一个batch维度
    # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    # grayscale_cam = cam(input_tensor=input_tensor)
    #
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
    #                                   grayscale_cam,
    #                                   use_rgb=True)
    # plt.imshow(visualization)
    # plt.savefig('./result.png')  # 将热力图的结果保存到本地当前文件夹
    # plt.show()
    img_dir = r"D:\BaiduNetdiskDownload\cohn-kanade-images\S010\005"  # 图片所在文件夹路径
    result_dir = "./results_ck"  # 结果保存文件夹路径
    image_size = 128  # 图像尺寸
    for filename in os.listdir(img_dir):  # 遍历文件夹中的所有图片
        img_path = os.path.join(img_dir, filename)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = center_crop_img(img, image_size)

        # [C, H, W]
        img_tensor = data_transform(img) if data_transform is not None else img
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        result_path = os.path.join(result_dir, filename.replace(".jpg", "_result.jpg"))
        plt.savefig(result_path)
        plt.close()


if __name__ == '__main__':
    main()
