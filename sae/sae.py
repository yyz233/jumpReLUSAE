from torch import Tensor, nn
import torch
from typing import NamedTuple
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
bandwidth = 0.001

def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).type_as(x)


class ForwarOutput(NamedTuple):
    sae_out: Tensor

    reconstruct_loss: Tensor
    """reconstruct loss"""

    sparsity_loss_without_coefficient: Tensor
    """sparsity loss"""


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return (x > threshold).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_grad = torch.zeros_like(x)
        threshold_grad = -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        return x_grad, threshold_grad


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return x * (x > threshold).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold).type_as(x) * grad_output
        threshold_grad = -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        return x_grad, threshold_grad


class JumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_in: int, 
        d_sae: int,
        device
    ):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device = device
        self.log_threshold = nn.Parameter(torch.zeros(d_sae))

    def set_log_threshold(self):
        self.log_threshold = nn.Parameter(torch.log(self.threshold))

    def forward(self, x):
        pre_activations = F.relu(x @ self.W_enc + self.b_enc)
        threshold = torch.exp(self.log_threshold)
        feature_magnitudes = JumpReLUFunction.apply(pre_activations, threshold)
        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec
        return x_reconstructed, feature_magnitudes

def load_jump_relu_sae_from_hub(repo_id, filename):
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}  
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1], 'cuda').to('cuda')
    sae.load_state_dict(pt_params, strict=False)
    sae.set_log_threshold()
    return sae