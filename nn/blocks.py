from typing import Callable, Optional, Union

from torch import Tensor, concat
from torch.nn import AvgPool2d, Module, ReLU, Sequential
from torch.nn.functional import avg_pool2d
from torch_geometric.typing import Adj

from nn.layers import Interpolate, PG2DLayer, conv_module
from nn.utils import KEEP_SIZE, conv_params


class GBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_module: Module,
        spectral_norm: bool = True,
        activation: Union[Module, Callable] = ReLU(inplace=False),
        upsample: Optional[Union[Module, Callable]] = Interpolate(scale_factor=2),
        simple: bool = False,
    ):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.spectral_norm = spectral_norm
        self.norm_module = norm_module
        self.activation = activation
        self.upsample = upsample
        self.simple = simple
        # Conv layers
        self.conv1 = conv_module(
            spectral_norm=self.spectral_norm,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **conv_params[KEEP_SIZE],
        )
        self.bn1 = self.norm_module(in_channels)
        if not self.simple:
            self.conv2 = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                **conv_params[KEEP_SIZE],
            )
            self.bn2 = self.norm_module(out_channels)
        self.learnable_sc = (
            self.in_channels != self.out_channels or self.upsample is not None
        )
        if self.learnable_sc:
            self.conv_sc = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )

    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        if not self.simple:
            h = self.activation(self.bn2(h))
            h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class PGGBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layout_channels: int,
        norm_module: Module,
        spectral_norm: bool = True,
        activation: Union[Module, Callable] = ReLU(inplace=False),
        upsample: Optional[Union[Module, Callable]] = Interpolate(scale_factor=2),
    ):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.layout_channels = layout_channels
        self.spectral_norm = spectral_norm
        self.norm_module = norm_module
        self.activation = activation
        self.upsample = upsample
        self.hidden_channels = self.out_channels

        # Conv layers
        self.conv1 = conv_module(
            spectral_norm=self.spectral_norm,
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            **conv_params[KEEP_SIZE],
        )
        self.conv2 = conv_module(
            spectral_norm=self.spectral_norm,
            in_channels=self.hidden_channels * 2,
            out_channels=self.out_channels,
            **conv_params[KEEP_SIZE],
        )
        self.layout_conv = conv_module(
            spectral_norm=self.spectral_norm,
            in_channels=self.layout_channels,
            out_channels=self.hidden_channels,
            **conv_params[KEEP_SIZE],
        )
        self.learnable_sc = (
            self.in_channels != self.out_channels or self.upsample is not None
        )
        if self.learnable_sc:
            self.conv_sc = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )
        # Batch-norm layers
        self.bn1 = self.norm_module(self.in_channels)
        self.bn2 = self.norm_module(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, layout, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        # Pooling after eventual upsample
        pooled_layout = avg_pool2d(layout, int(layout.shape[-1] / h.shape[-1]))
        # TODO maybe normalization and activation should be also here
        k = self.layout_conv(pooled_layout)
        h = self.conv2(concat((h, k), dim=1))
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class DBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spectral_norm: bool = True,
        wide: bool = True,
        preactivation: bool = True,
        activation: Union[Module, Callable] = ReLU(inplace=False),
        downsample: Optional[Union[Module, Callable]] = AvgPool2d(2),
        simple: bool = False,
    ):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.spectral_norm = spectral_norm
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
        self.simple = simple

        if self.simple:
            self.conv = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **conv_params[KEEP_SIZE],
            )
        else:
            # Conv layers
            self.conv1 = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                **conv_params[KEEP_SIZE],
            )
            self.conv2 = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.hidden_channels,
                out_channels=self.out_channels,
                **conv_params[KEEP_SIZE],
            )
        self.learnable_sc = (
            self.in_channels != self.out_channels or self.downsample is not None
        )
        if self.learnable_sc:
            self.conv_sc = conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample is not None:
                x = self.downsample(x)
        else:
            if self.downsample is not None:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.simple:
            h = self.conv(self.activation(x))
        else:
            if self.preactivation:
                h = self.activation(x)
            else:
                h = x
            h = self.conv1(h)
            h = self.conv2(self.activation(h))
        if self.downsample is not None:
            h = self.downsample(h)
        return h + self.shortcut(x)


class PG2DBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_module: Module,
        activation: Union[Module, Callable] = ReLU(inplace=False),
        spectral_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm
        self.norm_module = norm_module
        self.activation = activation
        nn = GBlock(
            in_channels=2 * self.in_channels,
            out_channels=self.out_channels,
            norm_module=self.norm_module,
            spectral_norm=self.spectral_norm,
            activation=self.activation,
            simple=True,
        )
        sc = Sequential(
            Interpolate(scale_factor=2),
            conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            ),
        )
        post_gating = conv_module(
            spectral_norm=self.spectral_norm,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.pg_layer = PG2DLayer(nn=nn, sc=sc, post_gating=post_gating)

    def forward(self, x: Tensor, edge_index: Adj):
        return self.pg_layer(x, edge_index)
