from typing import List, Optional

import torch.nn as nn
import stribor as st
from torch import Tensor
from torch.nn import Module


class CouplingFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = st.Flow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        return self.flow(x, t=t)[0]


class ResNetFlow(Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        invertible: Optional[bool] = True,
        **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(st.net.ResNetFlow(
                dim,
                hidden_dims,
                n_layers,
                activation='ReLU',
                final_activation=None,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
                invertible=invertible
            ))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        for layer in self.layers:
            x = layer(x, t)

        return x

class GruFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: list,
        time_net: str = None, 
        time_hidden_dim: int = None,
    ):
        super().__init__()
        self.gru_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gru_layers.append(
                nn.GRU(input_size=dim, hidden_size=hidden_dims[-1], batch_first=True)
            )
        self.output_layer = nn.Linear(hidden_dims[-1], dim)

    def forward(self, x, t):
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)

        x = self.output_layer(x)
        return x
    
class MLPFlow(nn.Module):
    def __init__(self, dim, n_layers, hidden_dims, activation="ReLU", **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        act_fn = getattr(nn, activation)
        for _ in range(n_layers):
            self.layers.append(nn.Linear(dim, hidden_dims[-1]))
            self.layers.append(act_fn())
            self.layers.append(nn.Linear(hidden_dims[-1], dim))

    def forward(self, x, t):
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x)
        return x