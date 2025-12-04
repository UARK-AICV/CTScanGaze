import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import pi
import math

epsilon = 1e-7 #regularization value in Keras

def CrossEntropyLoss(input, gt, mask):
    batch, time_scale, action = input.size()
    input = F.softmax(input, dim=-1)
    loss = -(gt * torch.log(input + epsilon) * mask.unsqueeze(-1)).sum() / mask.sum()
    return loss

def MLPLogNormalDistribution(log_normal_mu, log_normal_sigma2, gt, mask):
    """
    Negative log-likelihood loss for log-normal distribution.

    Log-normal PDF: p(t) = \frac{1}{t\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln t - \mu)^2}{2\sigma^2}\right)
    Log PDF: \log p(t) = -\log t - \log\sigma - \frac{1}{2}\log(2\pi) - \frac{(\ln t - \mu)^2}{2\sigma^2}
    """
    # batch, time_scale = log_normal_mu.size()
    logpdf = torch.log(1 / (gt + epsilon) * 1 / (torch.sqrt(2 * math.pi * log_normal_sigma2))) \
             + (- (torch.log(gt + epsilon) - log_normal_mu) ** 2 / (2 * log_normal_sigma2))
    loss = (logpdf[mask == 1]).sum() / mask.sum()
    return -loss