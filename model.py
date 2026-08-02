import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

# goal is to swap this simple net into a proper resnet

# the main convolution blocks needed
def conv_block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool = False, activate=True):
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
    ]

    # allows us to not relu on all convs
    # as relu can lead to loosing important data
    # in this case in conv2 the relu is held back until after the skip-add in the residual block
    if activate:
        # what does it do
        # relu is a non linear activation function that introduces non linearity to the mode;
        # non-linearity: keeps positive values unchanged, zeroes out negatives
        # without this, stacked conv layers would mathematically collapse into
        # one single linear operation regardless of depth — ReLU is what lets
        # the network represent complex, non-linear patterns
        layers.append(nn.ReLU(inplace = True))
    if pool:
        # pooling means literally applying a filter that shrinks the image
        # here it takes the max value in a 2x2 window and moves the window by 2 pxls
        # downsampling forces the model to learn more abstract features and reduces computation
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    # dropout blocks out randomly selected neurons so that neurons are less likely to be codependant
    # instead they will be forced to geenralize and have meaningful stand on their own
    # this significantly reduces overfitting and improves generalization
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = conv_block(in_channels, out_channels, stride=stride)
        # dropout2d blocks entire channels/feature_maps instead of individual pixels as pixels alone are not meaningful after convolutioin their effect gets easily lost
        # the if statement is to avoid creating a dropout layer is the dropout is 0 since it will be useless overhead
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv_block(out_channels, out_channels, stride=1, activate=False)
        # why did we not use activate = true instead of relu here?
        # in order to implement skip connection
        # ie, adding the input back into the output of the block,
        # we need to keep the output of conv2 linear (no activation) so that the addition is valid.
        # The ReLU activation is applied after the addition, allowing the network to learn residual mappings effectively.
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_block(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        # dropout is applied between the 2 convolutions
        # the 1st conv extracts features
        # the 2nd combines them
        # dropout randomly removes some of the features to force the model to learn more robust features
        # rather than highly codependant ones
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = conv_block(3, 64)  # 32x32 -> 32x32, no downsampling yet, just learns first filters

        # each stage = 2 ResidualBlocks. first block in a stage handles the
        # channel/size change (stride=2 shrinks spatial size, except stage 1),
        # second block just deepens at the new shape (stride=1, same channels)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # collapses whatever spatial size remains down to 1x1
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x