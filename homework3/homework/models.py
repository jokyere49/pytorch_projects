import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputCovLayer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.convBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.convBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.outputConvLayer = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Conv2d(128, 6, kernel_size=1, stride=1)
        )

        self.identity1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(64)
        )

        self.identity2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(128)
        )

    def forward(self, x):
        x = self.inputCovLayer(x)
        x = self.convBlock1(x) + self.identity1(x)
        x = self.convBlock2(x) + self.identity2(x)
        x = self.outputConvLayer(x)
        return x.mean(dim=[2, 3])


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.convBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.convBlock3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.identity2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(64)
        )

        self.identity3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(128)
        )

        self.upConvBlock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.upConvBlock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2*64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.upConvBlock3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2*32, 5, kernel_size=7, padding=3, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.convBlock1(x)
        x2 = self.convBlock2(x1) + self.identity2(x1)
        x3 = self.convBlock3(x2) + self.identity3(x2)
        x4 = self.upConvBlock1(x3)

        dim2IncreaseSkip1 = 0
        dim3IncreaseSkip1 = 0
        dim2IncreaseSkip2 = 0
        dim3IncreaseSkip2 = 0

        if x2.shape[2] == x4.shape[2] and x2.shape[3] == x4.shape[3]:
            x5 = self.upConvBlock2(torch.cat((x4, x2), dim=1))

        elif x2.shape[2] > x4.shape[2] or x2.shape[3] > x4.shape[3]:
            if x2.shape[2] > x4.shape[2]:
                dim2IncreaseSkip1 = x2.shape[2]-x4.shape[2]
            if x2.shape[3] > x4.shape[3]:
                dim3IncreaseSkip1 = x2.shape[3]-x4.shape[3]
            x4 =F.pad(input=x4, pad=(0, dim3IncreaseSkip1, 0, dim2IncreaseSkip1), mode='constant', value=0)
            x5 = self.upConvBlock2(torch.cat((x4, x2), dim=1))

        elif x2.shape[2] < x4.shape[2] or x2.shape[3] < x4.shape[3]:

            if x2.shape[2] < x4.shape[2]:
                dim2IncreaseSkip1 = x4.shape[2]-x2.shape[2]

            if x2.shape[3] < x4.shape[3]:
                dim3IncreaseSkip1 = x4.shape[3]-x2.shape[3]

            x2 =F.pad(input=x2, pad=(0, dim3IncreaseSkip1, 0, dim2IncreaseSkip1), mode='constant', value=0)

            x5 = self.upConvBlock2(torch.cat((x4, x2), dim=1))

        if x1.shape[2] == x5.shape[2] and x1.shape[3] == x5.shape[3]:
            x6 = self.upConvBlock3(torch.cat((x5, x1), dim=1))

        elif x1.shape[2] > x5.shape[2] or x1.shape[3] > x5.shape[3]:
            if x1.shape[2] > x5.shape[2]:
                dim2IncreaseSkip2 = x1.shape[2]-x5.shape[2]
            if x1.shape[3] > x5.shape[3]:
                dim3IncreaseSkip2 = x1.shape[3]-x5.shape[3]
            x5 =F.pad(input=x5, pad=(0, dim3IncreaseSkip2, 0, dim2IncreaseSkip2), mode='constant', value=0)
            x6 = self.upConvBlock3(torch.cat((x5, x1), dim=1))

        elif x1.shape[2] < x5.shape[2] or x1.shape[3] < x5.shape[3]:
            if x1.shape[2] < x5.shape[2]:
                dim2IncreaseSkip2 = x5.shape[2]-x1.shape[2]

            if x1.shape[3] < x5.shape[3]:
                dim3IncreaseSkip2 = x5.shape[3]-x1.shape[3]

            x1 =F.pad(input=x1, pad=(0, dim3IncreaseSkip2, 0, dim2IncreaseSkip2), mode='constant', value=0)

            x6 = self.upConvBlock3(torch.cat((x5, x1), dim=1))

        if x6.shape[2] == x.shape[2] and x6.shape[3] == x.shape[3]:
            return x6
        else:
            return x6[:, :, :x.shape[2], :x.shape[3]]


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
