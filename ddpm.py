import torch
from torch import Tensor

def sample_sigma(n: int, loc: float=-1.2, scale: float=1.2, sigma_min: float=2e-3, sigma_max: float=80, device=None):
    return (torch.randn(n, device=device) * scale + loc).exp().clip(sigma_min, sigma_max)

def c_in(sigma: Tensor, sigma_data: Tensor):
    return (sigma_data.pow(2) + sigma.pow(2)).pow(-1/2)

def c_out(sigma: Tensor, sigma_data: Tensor):
    return sigma * sigma_data * (sigma_data.pow(2) + sigma.pow(2)).pow(-1/2)

def c_skip(sigma: Tensor, sigma_data: Tensor):
    return sigma_data.pow(2) / (sigma_data.pow(2) + sigma.pow(2))

def c_noise(sigma: Tensor, sigma_data: Tensor):
    return sigma.log() / 4

def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

