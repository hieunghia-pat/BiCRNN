from torch import nn 

class Encoder(nn.Module):
    def __init__(self, imgChannels):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            self.create_cnn(imgChannels, 16, maxpool=True, dropout=0),
            self.create_cnn(16, 32, maxpool=True, dropout=0),
            self.create_cnn(32, 48, maxpool=True, dropout=0.2),
            self.create_cnn(48, 64, maxpool=False, dropout=0.2),
            self.create_cnn(64, 80, maxpool=False, dropout=0.2)
        )

    def create_cnn(self, inChannels, outChannels, kernel=3, stride=1, maxpool=True, dropout=0.2):
        if maxpool:
            return nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=kernel, stride=stride),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(outChannels),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )
        else: 
            return nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=kernel, stride=stride),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(outChannels),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.layers(x)

        return x