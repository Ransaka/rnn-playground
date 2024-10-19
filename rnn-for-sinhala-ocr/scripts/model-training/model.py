import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_channels, hidden_size, num_classes):
        super(CRNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=(2,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=(2,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.rnn = nn.LSTM(num_channels * 2 * 16, hidden_size, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]

        # CNN feature extraction
        conv = self.conv1(x)
        conv = self.conv2(conv)
        batch, channels, height, width = conv.size()

        conv = conv.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        conv = conv.contiguous().view(batch, width, channels * height)

        rnn, _ = self.rnn(conv)

        output = self.fc(rnn)

        return output