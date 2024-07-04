from typing import Callable, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, sigmoid
from torch.nn import Conv2d, Embedding, Identity, Linear, ModuleList
from torch.nn import Parameter as P
from torch.nn import Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, PairTensor, Size
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor, set_diag


class Interpolate(torch.nn.modules.Module):
    def __init__(self, **kwargs):
        super(Interpolate, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, **self.kwargs)
        return x


def sn_module(
    spectral_norm: bool, module_type: Type[torch.nn.Module], *args, **kwargs
) -> torch.nn.Module:
    module = module_type(*args, **kwargs)
    if spectral_norm:
        module = torch.nn.utils.parametrizations.spectral_norm(module=module)
    return module


def conv_module(spectral_norm: bool = False, *args, **kwargs) -> torch.nn.Module:
    return sn_module(spectral_norm=spectral_norm, module_type=Conv2d, *args, **kwargs)


def linear_module(spectral_norm: bool = False, *args, **kwargs) -> torch.nn.Module:
    return sn_module(spectral_norm=spectral_norm, module_type=Linear, *args, **kwargs)


# Attention code from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
class Attention(torch.nn.Module):
    def __init__(self, ch: int, spectral_norm: bool, conv: Callable = conv_module):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.conv = conv
        self.theta = self.conv(
            spectral_norm=spectral_norm,
            in_channels=self.ch,
            out_channels=self.ch // 8,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
        )
        self.phi = self.conv(
            spectral_norm=spectral_norm,
            in_channels=self.ch,
            out_channels=self.ch // 8,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
        )
        self.g = self.conv(
            spectral_norm=spectral_norm,
            in_channels=self.ch,
            out_channels=self.ch // 2,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
        )
        self.o = self.conv(
            spectral_norm=spectral_norm,
            in_channels=self.ch // 2,
            out_channels=self.ch,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
        )
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(
                -1, self.ch // 2, x.shape[2], x.shape[3]
            )
        )
        return self.gamma * o + x


class GPoseConv(MessagePassing):
    r"""GPose Layer

    .. math::
        \mathbf{x}^{\prime}_i = g_{\mathbf{\Theta}} \left( k_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{p}_i \right) +
        \sum_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}} ( \mathbf{x}_j,
        \mathbf{p}_j - \mathbf{p}_i) \right) \right),

    where :math:`\g_{\mathbf{\Theta}}`, :math:`\k_{\mathbf{\Theta}}`
    and :math:`h_{\mathbf{\Theta}}`
    denote neural networks, *i.e.* MLPs, and
    :math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
    each point.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`k_{\mathbf{\Theta}}` that maps node features :obj:`x` and
            spatial coordinates :obj:`pos_i` of shape
            :obj:`[-1, in_channels + num_dimensions]` to shape
            :obj:`[-1, hidden_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
            relative spatial coordinates :obj:`pos_j - pos_i` of shape
            :obj:`[-1, in_channels + num_dimensions]` to shape
            :obj:`[-1, hidden_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\g_{\mathbf{\Theta}}` that maps aggregated node features
            of shape :obj:`[-1, hidden_channels + in channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          positions :math:`(|\mathcal{V}|, 3)` or
          :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        self_nn: Optional[Callable] = None,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        add_self_loops: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.self_nn = self_nn
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.self_nn)
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(
        self,
        x: Union[OptTensor, OptPairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = add_remaining_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0))
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        x_r = x[1]
        pos_r = pos[1]
        if x_r is not None and self.self_nn is not None:
            out += self.self_nn(torch.cat([x_r, pos_r], dim=1))

        if self.global_nn is not None:
            out = self.global_nn(torch.cat([out, x_r], dim=1))

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(self_nn={self.self_nn}, "
            f"local_nn={self.local_nn}, "
            f"global_nn={self.global_nn})"
        )


class GPoseLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_dimensions: int,
        spectral_norm: bool = True,
        skip_connection: bool = True,
    ):
        super().__init__()
        self.spectral_norm = spectral_norm
        self.skip_connection = skip_connection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_dimensions = num_dimensions
        self_nn = linear_module(
            spectral_norm=self.spectral_norm,
            in_features=self.in_channels + self.num_dimensions,
            out_features=self.hidden_channels,
        )
        local_nn = linear_module(
            spectral_norm=self.spectral_norm,
            in_features=self.in_channels + self.num_dimensions,
            out_features=self.hidden_channels,
        )

        global_nn = linear_module(
            spectral_norm=self.spectral_norm,
            in_features=self.hidden_channels + self.in_channels,
            out_features=self.out_channels,
        )
        self.learnable_sc = (
            self.skip_connection and self.in_channels != self.out_channels
        )
        if self.learnable_sc:
            self.sc = linear_module(
                spectral_norm=self.spectral_norm,
                in_features=self.in_channels,
                out_features=self.out_channels,
            )
        self.pose_conv = GPoseConv(
            self_nn=Sequential(self_nn),
            local_nn=Sequential(local_nn),
            global_nn=Sequential(global_nn),
        )

    def forward(self, x, pos, edge_index):
        if self.skip_connection:
            if self.learnable_sc:
                out = self.sc(x) + self.pose_conv(x, pos, edge_index)
            else:
                out = x + self.pose_conv(x, pos, edge_index)
        else:
            out = self.pose_conv(x, pos, edge_index)
        return out


class BaselineLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dimensions: int,
        spectral_norm: bool = True,
        skip_connection: bool = True,
    ):
        super().__init__()
        self.spectral_norm = spectral_norm
        self.skip_connection = skip_connection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dimensions = num_dimensions
        self.self_nn = linear_module(
            spectral_norm=self.spectral_norm,
            in_features=self.in_channels + self.num_dimensions,
            out_features=self.out_channels,
        )
        self.learnable_sc = (
            self.skip_connection and self.in_channels != self.out_channels
        )
        if self.learnable_sc:
            self.sc = linear_module(
                spectral_norm=self.spectral_norm,
                in_features=self.in_channels,
                out_features=self.out_channels,
            )

    def forward(self, x, pos):
        if self.skip_connection:
            if self.learnable_sc:
                out = self.sc(x) + self.self_nn(torch.cat([x, pos], dim=1))
            else:
                out = x + self.self_nn(torch.cat([x, pos], dim=1))
        else:
            out = self.self_nn(torch.cat([x, pos], dim=1))
        return out


class PG2DLayer(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        sc: Callable,
        post_gating: Optional[Callable] = None,
        **kwargs,
    ):
        """Consider your data to be of shape [N,C,H1,W1]

        :param nn: A NN that takes
        inputs of shape [N,2*C,H1,W1] and outputs shape [N,O,H2,W2]
        :param sc: A learnable skip connection that takes
        inputs of shape [N,C,H1,W1] and outputs shape [N,O,H2,W2]
        :param kwargs:
        """
        super().__init__(node_dim=-4, **kwargs)
        self.nn = nn
        self.sc = sc
        self.post_gating = post_gating if post_gating is not None else Identity()
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.sc)
        reset(self.post_gating)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.sc(x) + sigmoid(out) * self.post_gating(out)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = self.nn(torch.cat((x_i, x_j), dim=1))
        return sigmoid(msg) * msg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn}, sc={self.sc})"


class MultiEmbedding(torch.nn.Module):
    def __init__(
        self, num_embeddings: Sequence[int], embedding_dim: int, aggr: str = "mean"
    ):
        super().__init__()
        self.embeddings = ModuleList(
            [
                Embedding(num_embeddings=n, embedding_dim=embedding_dim)
                for n in num_embeddings
            ]
        )
        if aggr == "mean":
            aggr = torch.mean
        elif aggr == "sum":
            aggr = torch.sum
        elif aggr == "max":
            aggr = torch.max
        elif aggr == "cat":
            aggr = torch.cat
        else:
            raise ValueError(f"aggr must be one of cat, mean, sum or max. Got {aggr}")
        self.aggr = aggr
        self.out_features = (
            embedding_dim * len(num_embeddings) if aggr == torch.cat else embedding_dim
        )

    def aggregate(self, embeddings: Sequence[Tensor]) -> Tensor:
        assert embeddings[0].ndim == 3, "MultiEmbedding aggregation assumes 3D Tensor"
        if self.aggr == torch.cat:
            result = self.aggr(embeddings, dim=1)
            result = torch.reshape(
                result, (result.shape[0], result.shape[1] * result.shape[2])
            )
        else:
            result = self.aggr(torch.cat(embeddings, dim=1), dim=1)
        return result

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        assert len(xs) == len(self.embeddings)
        # [n,f] -> [n,f,e]
        embeddings = [embedding(x) for embedding, x in zip(self.embeddings, xs)]
        # [n,f,e] -> [n,e] | [n,f*e]
        return self.aggregate(embeddings)


class MultiMaskedEmbedding(MultiEmbedding):
    def forward(
        self, xs: Sequence[Tensor], masks: Optional[Sequence[Tensor]] = None
    ) -> Tensor:
        if masks is None:
            return super().forward(xs)
        assert len(xs) == len(self.embeddings) == len(masks)
        embeddings = []
        for embedding, x, mask in zip(self.embeddings, xs, masks):
            e = embedding(x)
            e[~mask] = 0.0
            embeddings.append(e)
        return self.aggregate(embeddings)
