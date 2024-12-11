import torch.nn as nn

from loss import mse_loss
from datasets import SpectrogramDataset
    
class SpectrVelCNNRegr(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.

    Number of parameters in model SpectrVelCNNRegr: 38414929 = 3.84e+07
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)
    
class SpectrVelCNNRegr_w_dropout(SpectrVelCNNRegr):
    """
        Added dropout layers for regularization to the linear layers, while still matching the 
        baseline architecture.
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=37120, out_features=1024),
            nn.Dropout(p=dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(p=dropout_rate)
        )
        self.linear3 = nn.Linear(in_features=256, out_features=1) 

    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

class SpectrVelCNNRegr_w_dropout_extra_CNN(SpectrVelCNNRegr):
    """Added dropout layers for regularization and a extra CNN layer."""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 16x36x459
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 32×18×229
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 64x9x114
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 128×5×58
        )


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 256×3×29
        )

        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=23040, out_features=1024),
            nn.Dropout(p=dropout_rate)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(p=dropout_rate)
        )
        self.linear3 = nn.Linear(in_features=256, out_features=1) 

    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
