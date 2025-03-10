from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from torch import nn
import ot
from torch import nn


def sample_gumbel(shape, eps=1e-10, device="cuda"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size(), device=logits.device)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (
        torch.sum(z_continuous_flat**2, dim=1, keepdim=True)
        + torch.sum(codebook**2, dim=1)
        - 2 * torch.matmul(z_continuous_flat, codebook.t())
    )

    return distances


class VectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=1, beta=1e-3):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
        self.beta = beta
        print("Using VectorQuantizer")

    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._quantize(z_from_encoder, codebook, flg_train)

    def _quantize(self, z, codebook, flg_train):
        # z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_flattened, codebook.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.size_dict).to(
            z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)
        if flg_train:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )
            # loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = 0.0
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        assert z_q.shape == z.shape

        return z_q, loss, perplexity

    def _inference(self, z, codebook):

        # z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_flattened, codebook.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.size_dict).to(
            z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.view(z.shape)
        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, min_encodings, min_encoding_indices, perplexity

    def set_temperature(self, value):
        self.temperature = value


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("EnsembleLinear") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class WSVectorQuantizer(VectorQuantizer):
    def __init__(
        self,
        size_dict,
        dim_dict,
        temperature=1,
        beta: float = 1e-3,
        kl_regularization: float = 1,
        global_optimization: bool = True,
        fixed_weight: bool = False,
        init_weights=None,
    ):
        super(WSVectorQuantizer, self).__init__(
            size_dict, dim_dict, temperature=temperature, beta=beta
        )

        self.kl_loss = nn.KLDivLoss()
        self.kl_regularization = kl_regularization
        self.global_optimization = global_optimization
        self.fixed_weight = fixed_weight
        self.init_weights = init_weights
        self.softmax = nn.Softmax(dim=0)
        print("---------------------------------------------------")
        print("Using WSVectorQuantizer")
        print("fixed_weight ", self.fixed_weight)
        print("global_optimization ", self.global_optimization)
        print("beta ", self.beta)
        print("kl_regularization ", self.kl_regularization)
        print("---------------------------------------------------")

    def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
        return self._quantize(z_from_encoder, codebook, codebook_weight, flg_train)

    def _quantize(self, z, codebook, codebook_weight, flg_train):
        # z = z.permute(2, 3, 0, 1).contiguous()

        z_flattened = z.view(-1, self.dim_dict)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        cost_matrix = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_flattened, codebook.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.size_dict).to(
            z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)

        if flg_train:
            if self.global_optimization:
                size_batch = z_flattened.shape[0]
                sample_weight = torch.ones(size_batch).to(z.device) / size_batch

                if self.fixed_weight:
                    loss = self.beta * ot.emd2(
                        codeword_weight,
                        sample_weight,
                        cost_matrix.t(),
                        numItermax=500000,
                    )
                else:
                    # for regularization
                    codeword_weight = (
                        torch.ones(self.size_dict).to(z.device) / self.size_dict
                    )
                    weight = self.softmax(codebook_weight.t())
                    loss = self.beta * ot.emd2(
                        weight, sample_weight, cost_matrix.t(), numItermax=500000
                    )
                    if self.kl_regularization > 0.0:
                        loss += self.kl_regularization * self.kl_loss(
                            codeword_weight.log(), weight
                        )

            else:

                loss = 0.0
                size_batch = z.shape[2]
                num_iter = z.shape[0] * z.shape[1]

                sample_weight = torch.ones(size_batch).to(z.device) / size_batch

                if self.fixed_weight:
                    if self.init_weights is None:
                        codeword_weight = (
                            torch.ones(self.size_dict).to(z.device) / self.size_dict
                        )
                    else:
                        codeword_weight = (
                            nn.Softmax(dim=0)(
                                torch.ones(self.size_dict) * self.init_weights
                            )
                        ).to(z.device)

                    for i in range(num_iter):
                        loss += self.beta * ot.emd2(
                            codeword_weight,
                            sample_weight,
                            cost_matrix[size_batch * i : size_batch * (i + 1)].t(),
                        )
                else:
                    codeword_weight = (
                        torch.ones(self.size_dict).to(z.device) / self.size_dict
                    )
                    weight = nn.Softmax(dim=1)(codebook_weight)

                    for i in range(num_iter):
                        loss += self.beta * ot.emd2(
                            weight[i],
                            sample_weight,
                            cost_matrix[size_batch * i : size_batch * (i + 1)].t(),
                        )
                        if self.kl_regularization > 0.0:
                            loss += self.kl_regularization * self.kl_loss(
                                codeword_weight.log(), weight[i]
                            )

        else:
            loss = 0.0

        z_q = z_q.view(z.shape)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(2, 3, 0, 1).contiguous()
        assert z_q.shape == z.shape

        return z_q, loss, perplexity


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=1, param_var_q="gaussian_1"):
        super(GaussianVectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
        self.param_var_q = param_var_q

    def forward(
        self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False
    ):
        return self._quantize(
            z_from_encoder,
            param_q,
            codebook,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
        )

    def _quantize(
        self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False
    ):
        # bs, dim_z, width, height = z_from_encoder.shape
        # z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        z_from_encoder_permuted = z_from_encoder

        bs, dim_z = z_from_encoder.shape

        precision_q = (1.0 / torch.clamp(var_q, min=1e-10)).to(z_from_encoder.device)
        logit = -self._calc_distance_bw_enc_codes(
            z_from_encoder_permuted, codebook, 0.5 * precision_q
        )
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)

        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(
                    indices.shape[0], self.size_dict, device=z_from_encoder.device
                )
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(
                    codebook
                )
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(
                bs, width, height, dim_z
            )
        # z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        z_to_decoder = z_quantized

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0, 1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(
            z_from_encoder, z_to_decoder, 0.5 * precision_q
        ).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity

    def _inference(
        self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False
    ):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1.0 / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(
            z_from_encoder_permuted, codebook, 0.5 * precision_q
        )

        min_encoding_indices = torch.argmax(logit, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.size_dict, device="cuda"
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_quantized = torch.matmul(min_encodings, codebook).view(
            bs, width, height, dim_z
        )
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0, 1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(
            z_from_encoder, z_to_decoder, 0.5 * precision_q
        ).mean()
        loss = kld_discrete + kld_continuous

        return z_to_decoder, min_encodings, min_encoding_indices, loss

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):
        distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        return distances

    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1 - x2) ** 2 * weight, dim=(-1))

    def set_temperature(self, value):
        self.temperature = value
