import torch
import tsai.all as tsai


class AlexNet(torch.nn.Module):

    def __init__(self, in_width, in_channels, num_classes):
        super(AlexNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(96),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            torch.nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            torch.nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(384),

            torch.nn.Conv1d(384, 384, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(384),

            torch.nn.Conv1d(384, 256, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),

            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        in_width = ((in_width-1)//4)+1  # conv1
        in_width = ((in_width-1)//2)+1  # maxpool1

        in_width = in_width  # conv2
        in_width = ((in_width-1)//2)+1  # maxpool2

        in_width = in_width  # conv3, conv4, conv5
        in_width = ((in_width-1)//2)+1  # maxpool3

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(in_width*256, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FCN(torch.nn.Module):

    def __init__(self, in_width, channels, num_classes):
        super(FCN, self).__init__()

        in_width = ((in_width-1)//4)+1  # conv1

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 96, kernel_size=11, stride=4, padding=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(96),

            torch.nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),

            torch.nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(384),

            torch.nn.Conv1d(384, 384, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(384),

            torch.nn.Conv1d(384, 256, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),

            torch.nn.Conv1d(256, num_classes, kernel_size=1, stride=1, padding=0),
            torch.nn.MaxPool1d(kernel_size=in_width, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class FDN(torch.nn.Module):

    def __init__(self, in_width, channels, num_classes):
        super(FDN, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_width*channels, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LSTM(torch.nn.Module):

    def __init__(self, in_width, channels, num_classes):
        super(LSTM, self).__init__()

        self.lstm_stack = torch.nn.LSTM(input_size=channels, hidden_size=512,
                                  num_layers=2, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(in_width*(512*2), num_classes)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.lstm_stack(x)
        x = torch.flatten(x[0], 1)
        x = self.linear(x)
        return x


class Inception(tsai.InceptionTime):
    def __init__(self, in_width, channels, num_classes):
        super(Inception, self).__init__(c_in=channels, c_out=num_classes)