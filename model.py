import torch
import torch.nn as nn
import torch.nn.functional as F

### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            #nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_kernels: int, out_kernels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.subsample = stride > 1 or in_kernels != out_kernels

        self.conv1 = nn.Conv2d(
            in_kernels, out_kernels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_kernels)

        self.conv2 = nn.Conv2d(
            out_kernels, out_kernels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_kernels)

        if self.subsample:
            self.id_conv = nn.Conv2d(
                in_kernels, out_kernels,
                kernel_size=1, stride=stride, bias=False
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        res = self.id_conv(x) if self.subsample else x

        return F.relu(res + h)


class ResNet(nn.Module):
    def __init__(
        self,
        layers: tuple[int, ...] = (3, 4, 6, 3),
        kernels: tuple[int, ...] = (64, 128, 256, 512),
        channels_in: int = 3,
        dim_out: int = 1000,
    ):
        super().__init__()
        assert len(layers) == len(kernels)

        self.conv = nn.Conv2d(
            channels_in, kernels[0],
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(kernels[0])

        blocks = []
        for j, num_blocks in enumerate(layers):
            for i in range(num_blocks):
                k_in  = ([kernels[0]] + list(kernels))[j] if i == 0 else kernels[j]
                k_out = kernels[j]
                stride = 2 if i == 0 and j > 0 else 1
                blocks.append(ResNetBlock(k_in, k_out, stride=stride))
        self.layers = nn.Sequential(*blocks)

        self.fc = nn.Linear(kernels[-1], dim_out)

        self._init_head()

    def _init_head(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x


class _ResNetContainerBlock(nn.Module):
    """
    ResNetBlock with independently parameterisable conv1 and conv2 output channels.
    Used inside ResNetContainer after FedMA matching, where each layer may have a
    different number of matched neurons.
    The shortcut (id_conv) is created whenever stride > 1 or in_kernels != out_kernels,
    mirroring the condition in ResNetBlock.
    """
    def __init__(self, in_kernels: int, mid_kernels: int, out_kernels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_kernels, mid_kernels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_kernels)
        self.conv2 = nn.Conv2d(mid_kernels, out_kernels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_kernels)
        self.id_conv = (
            nn.Conv2d(in_kernels, out_kernels, kernel_size=1, stride=stride, bias=False)
            if stride > 1 or in_kernels != out_kernels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        res = self.id_conv(x) if self.id_conv is not None else x
        return F.relu(res + h)


class ResNetContainer(nn.Module):
    """
    Parameterisable ResNet for FedMA post-matching.

    num_filters : flat list of output-channel counts for every main-path conv layer,
                  in the order produced by pdm_prepare_full_weights_cnn.
                  Length must be  1 + 2 * sum(layers):
                    index 0          → initial stem conv
                    indices 1, 2     → stage-0 block-0  (conv1, conv2)
                    indices 3, 4     → stage-0 block-1  (conv1, conv2)
                    ...

    BN γ (weight) parameters are initialised to 1; the matching procedure only
    targets conv weights and BN β (bias).  γ is re-learned during local retraining.
    id_conv (shortcut projection) weights are initialised randomly and re-learned;
    they are reconstructed explicitly only in reconstruct_local_net.
    """
    def __init__(
        self,
        num_filters: list,
        layers: tuple = (3, 4, 6, 3),
        channels_in: int = 3,
        dim_out: int = 1000,
    ):
        super().__init__()
        assert len(num_filters) == 1 + 2 * sum(layers), (
            f"num_filters must have length 1+2*sum(layers)={1+2*sum(layers)}, got {len(num_filters)}")

        self.conv = nn.Conv2d(channels_in, num_filters[0],
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(num_filters[0])

        blocks = []
        f_idx = 1           # next position in num_filters
        in_ch = num_filters[0]
        for j, num_blocks in enumerate(layers):
            for i in range(num_blocks):
                mid_ch = num_filters[f_idx]
                out_ch = num_filters[f_idx + 1]
                stride = 2 if (i == 0 and j > 0) else 1
                blocks.append(_ResNetContainerBlock(in_ch, mid_ch, out_ch, stride=stride))
                in_ch = out_ch
                f_idx += 2

        self.layers = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x