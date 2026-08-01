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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = conv_block(channels, channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x  # the skip connection: add the original input back