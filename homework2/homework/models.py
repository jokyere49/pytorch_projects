import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.inputCovLayer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.convBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
        )
        self.convBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
        )
        self.outputConvLayer = torch.nn.Conv2d(128, 6, kernel_size=1, stride=1)
        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        x = self.inputCovLayer(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.outputConvLayer(x)
        return x.mean(dim=[2, 3])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
