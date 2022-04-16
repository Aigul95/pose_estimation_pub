import torch
import torch.nn as nn


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        name="conv1_1",
    ):
        super(ConvRelu, self).__init__()

        assert "conv" in name
        self.name = name
        self.add_module(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module(name.replace("conv", "relu"), nn.ReLU(inplace=True))

    def forward(self, x):
        conv = getattr(self, self.name)
        relu = getattr(self, self.name.replace("conv", "relu"))

        return relu(conv(x))


class ConvPRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        name="conv1_1",
    ):
        super(ConvPRelu, self).__init__()

        assert "conv" in name
        self.name = name
        self.add_module(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module(name.replace("conv", "prelu"), nn.PReLU(out_channels))

    def forward(self, x):
        conv = getattr(self, self.name)
        relu = getattr(self, self.name.replace("conv", "prelu"))

        return relu(conv(x))


class Pool(nn.Module):
    def __init__(self, kernel, stride, pad, num_pool, num_stage):
        super(Pool, self).__init__()

        self.num_pool, self.num_stage = num_pool, num_stage
        self.add_module(
            f"pool{num_pool}_stage{num_stage}", nn.MaxPool2d(kernel, stride=stride, padding=pad)
        )

    def forward(self, x):
        pool = getattr(self, f"pool{self.num_pool}_stage{self.num_stage}")
        return pool(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, num_stage, level=2):
        super(ConvBlock, self).__init__()

        self.num_conv = num_conv
        self.level = level
        self.num_stage = num_stage

        self.add_module(
            f"Mconv{num_conv}_stage{num_stage}_L{level}_0",
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.add_module(
            f"Mconv{num_conv}_stage{num_stage}_L{level}_1",
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.add_module(
            f"Mconv{num_conv}_stage{num_stage}_L{level}_2",
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        self.add_module(f"Mprelu{num_conv}_stage{num_stage}_L{level}_0", nn.PReLU(out_channels))

        self.add_module(f"Mprelu{num_conv}_stage{num_stage}_L{level}_1", nn.PReLU(out_channels))
        self.add_module(f"Mprelu{num_conv}_stage{num_stage}_L{level}_2", nn.PReLU(out_channels))

    def forward(self, x):
        prelu0 = getattr(self, f"Mprelu{self.num_conv}_stage{self.num_stage}_L{self.level}_0")
        conv0 = getattr(self, f"Mconv{self.num_conv}_stage{self.num_stage}_L{self.level}_0")

        x1 = prelu0(conv0(x))
        prelu1 = getattr(self, f"Mprelu{self.num_conv}_stage{self.num_stage}_L{self.level}_1")
        conv1 = getattr(self, f"Mconv{self.num_conv}_stage{self.num_stage}_L{self.level}_1")

        x2 = prelu1(conv1(x1))
        prelu2 = getattr(self, f"Mprelu{self.num_conv}_stage{self.num_stage}_L{self.level}_2")
        conv2 = getattr(self, f"Mconv{self.num_conv}_stage{self.num_stage}_L{self.level}_2")

        x3 = prelu2(conv2(x2))
        return torch.cat([x1, x2, x3], dim=1)


class SNWBPE(nn.Module):
    def __init__(self):
        super(SNWBPE, self).__init__()
        self.backbone = nn.Sequential(
            ConvRelu(3, 64, 3, 1, 1, name="conv1_1"),
            ConvRelu(64, 64, 3, 1, 1, name="conv1_2"),
            Pool(2, 2, 0, num_pool=1, num_stage=1),
            ConvRelu(64, 128, 3, 1, 1, name="conv2_1"),
            ConvRelu(128, 128, 3, 1, 1, name="conv2_2"),
            Pool(2, 2, 0, num_pool=2, num_stage=1),
            ConvRelu(128, 256, 3, 1, 1, name="conv3_1"),
            ConvRelu(256, 256, 3, 1, 1, name="conv3_2"),
            ConvRelu(256, 256, 3, 1, 1, name="conv3_3"),
            ConvRelu(256, 256, 3, 1, 1, name="conv3_4"),
            Pool(2, 2, 0, num_pool=3, num_stage=1),
            ConvRelu(256, 512, 3, 1, 1, name="conv4_1"),
            ConvPRelu(512, 512, 3, 1, 1, name="conv4_2"),
            ConvPRelu(512, 256, 3, 1, 1, name="conv4_3_CPM"),
            ConvPRelu(256, 128, 3, 1, 1, name="conv4_4_CPM"),
        )

        self.stage0 = nn.Sequential(
            ConvBlock(128, 128, num_conv=1, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=2, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=3, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=4, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=5, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=6, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=7, num_stage=0),
            ConvBlock(128 * 3, 128, num_conv=8, num_stage=0),
            ConvPRelu(128 * 3, 256, 1, 1, 0, name="Mconv9_stage0_L2"),
        )
        self.stage0.add_module("Mconv10_stage0_L2", nn.Conv2d(256, 304, 1, 1, 0))

        self.stage1 = nn.Sequential(
            ConvBlock(128 + 304, 128, num_conv=1, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=2, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=3, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=4, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=5, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=6, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=7, num_stage=1),
            ConvBlock(128 * 3, 128, num_conv=8, num_stage=1),
            ConvPRelu(128 * 3, 512, 1, 1, 0, name="Mconv9_stage1_L2"),
        )
        self.stage1.add_module("Mconv10_stage1_L2", nn.Conv2d(512, 304, 1, 1, 0))
        self.stage2 = nn.Sequential(
            ConvBlock(128 + 304, 128, num_conv=1, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=2, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=3, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=4, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=5, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=6, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=7, num_stage=2),
            ConvBlock(128 * 3, 128, num_conv=8, num_stage=2),
            ConvPRelu(128 * 3, 512, 1, 1, 0, name="Mconv9_stage2_L2"),
        )
        self.stage2.add_module("Mconv10_stage2_L2", nn.Conv2d(512, 304, 1, 1, 0))

        self.stage3 = nn.Sequential(
            ConvBlock(128 + 304, 256, num_conv=1, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=2, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=3, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=4, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=5, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=6, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=7, num_stage=3),
            ConvBlock(256 * 3, 256, num_conv=8, num_stage=3),
            ConvPRelu(256 * 3, 512, 1, 1, 0, name="Mconv9_stage3_L2"),
        )
        self.stage3.add_module("Mconv10_stage3_L2", nn.Conv2d(512, 304, 1, 1, 0))

        self.stage4 = nn.Sequential(
            ConvBlock(128 + 304, 256, num_conv=1, num_stage=0, level=1),
            ConvBlock(256 * 3, 256, num_conv=2, num_stage=0, level=1),
            ConvBlock(256 * 3, 256, num_conv=3, num_stage=0, level=1),
            ConvBlock(256 * 3, 256, num_conv=4, num_stage=0, level=1),
            ConvBlock(256 * 3, 256, num_conv=5, num_stage=0, level=1),
            ConvPRelu(256 * 3, 512, 1, 1, 0, name="Mconv6_stage0_L1"),
        )
        self.stage4.add_module("Mconv7_stage0_L1", nn.Conv2d(512, 135, 1, 1, 0))

    def forward(self, x):
        features = self.backbone(x)  # 128
        x1 = self.stage0(features)  # 304
        x2 = torch.cat([features, x1], dim=1)  # 432
        x3 = self.stage1(x2)
        x4 = torch.cat([features, x3], dim=1)
        x5 = self.stage2(x4)
        x6 = torch.cat([features, x5], dim=1)
        x7 = self.stage3(x6)
        x8 = torch.cat([features, x7], dim=1)
        x9 = self.stage4(x8)
        x7 = x7[:, :32, :, :]
        x9 = x9[:, :18, :, :]
        return x7, x9
