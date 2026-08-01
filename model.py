import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# goal is to swap this simple net into a proper resnet

# the main convolution blocks needed
def conv_block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool = False):
    layers = [
        # normal convolution layer with defined ios
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),

        # raw pixel values are widely varying and non consistent for a model to learn
        # normalizes each channel's values to mean 0, std 1 (not a fixed [0,1] range),
        # then applies learnable scale/shift (gamma, beta) — stabilizes the distribution
        # of activations flowing into the next layer, rather than compressing to a fixed range
        # this normalization is done per channel, so each channel is normalized independently
        # it helps the model learn real relationships by unifying the scale of the input data and reducing internal covariate shift
        # internal covariate shift is when the distribution of inputs to a layer
        # # changes during training, slowing down convergence and making optimization harder
        nn.BatchNorm2d(out_channels),

        # what does it do
        # relu is a non linear activation function that introduces non linearity to the mode;
        # non-linearity: keeps positive values unchanged, zeroes out negatives
        # without this, stacked conv layers would mathematically collapse into
        # one single linear operation regardless of depth — ReLU is what lets
        # the network represent complex, non-linear patterns
        nn.ReLU(inplace = True)
    ]
    if pool:
        # pooling means literally applying a filter that shrinks the image
        # here it takes the max value in a 2x2 window and moves the window by 2 pxls
        # downsampling forces the model to learn more abstract features and reduces computation
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    return nn.Sequential(*layers)

# Simple CNN for CIFAR-10
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Use einops for clarity
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = rearrange(x, 'b c h w -> b (c h w)')  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x