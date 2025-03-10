import numpy as np
import torch
import torch.nn

from .nnutils import Network, Reshape
from .quantizer import (
    WSVectorQuantizer,
    VectorQuantizer,
    GaussianVectorQuantizer,
)


class PhiNet(Network):
    def __init__(
        self,
        input_shape=2,
        n_latent_dims=4,
        n_hidden_layers=1,
        n_units_per_layer=32,
        final_activation=torch.nn.Tanh,
    ):
        super().__init__()
        self.input_shape = input_shape

        shape_flat = np.prod(self.input_shape)

        self.layers = []
        self.layers.extend([Reshape(-1, shape_flat)])
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(shape_flat, n_latent_dims)])
        else:
            self.layers.extend(
                [torch.nn.Linear(shape_flat, n_units_per_layer), torch.nn.Tanh()]
            )
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()]
                * (n_hidden_layers - 1)
            )
            self.layers.extend(
                [
                    torch.nn.Linear(n_units_per_layer, n_latent_dims),
                ]
            )
        if final_activation is not None:
            self.layers.extend([final_activation()])
        self.phi = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.phi(x)
        return z


class VectorQuantizedPhiNet(Network):
    """
    Custom Veector quantized Phi network,
    extend from the original Phi network
    with extra codebooks
    """

    def __init__(
        self,
        input_shape=2,
        n_latent_dims=4,
        n_hidden_layers=1,
        n_units_per_layer=32,
        final_activation=torch.nn.Tanh,
        codebook_size=100,
        use_vqwae: bool = False,
        use_sqvae: bool = False,
    ):
        super().__init__()
        self.input_shape = input_shape

        shape_flat = np.prod(self.input_shape)

        self.layers = []
        self.layers.extend([Reshape(-1, shape_flat)])
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(shape_flat, n_latent_dims)])
        else:
            self.layers.extend(
                [torch.nn.Linear(shape_flat, n_units_per_layer), torch.nn.Tanh()]
            )
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()]
                * (n_hidden_layers - 1)
            )
            self.layers.extend(
                [
                    torch.nn.Linear(n_units_per_layer, n_latent_dims),
                ]
            )

        # custom quantized network starts here
        self.size_dict = codebook_size
        self.dim_dict = n_latent_dims
        self.final_activation = final_activation

        self.codebook = torch.nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        # uniform weight codebook
        self.codebook_weight = torch.nn.Parameter(
            torch.ones(self.size_dict) / self.size_dict
        )
        self.log_param_q_scalar = torch.nn.Parameter(torch.tensor(2.995732273553991))

        if use_vqwae:
            self.quantizer = WSVectorQuantizer(self.size_dict, self.dim_dict)
        elif use_sqvae:
            self.quantizer = GaussianVectorQuantizer(self.size_dict, self.dim_dict)
        else:
            self.quantizer = VectorQuantizer(self.size_dict, self.dim_dict)

        if final_activation is not None:
            self.last_layer = final_activation()

        self.phi = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        z_from_encoder = self.phi(x)
        # flg_train is always True for now
        if isinstance(self.quantizer, GaussianVectorQuantizer):
            log_var_q = torch.tensor([0.0], device=x.device)
            self.param_q = log_var_q.exp() + self.log_param_q_scalar.exp()
            z_quantized, loss_latent, _ = self.quantizer(
                z_from_encoder, self.param_q, self.codebook, True, True
            )
        else:
            z_quantized, loss_latent, _ = self.quantizer(
                z_from_encoder, self.codebook, self.codebook_weight, True
            )
        if self.final_activation:
            z_quantized = self.last_layer(z_quantized)
        self.latent_loss = loss_latent
        # no decoder
        return z_quantized
