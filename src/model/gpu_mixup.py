import torch
from torch import nn
from torch.distributions import Beta


class Mixup(nn.Module):
    """
    A PyTorch module that implements Mixup data augmentation.

    Mixup is a data augmentation technique that creates virtual training examples
    by linear interpolation of both features and labels of random pairs of samples.

    Args:
        mix_beta (float): Parameter of the Beta distribution used for mixing coefficient sampling.
        mixadd (bool, optional): If True, adds the labels instead of interpolating them. Defaults to False.

    Methods:
        forward(x, y, z=None):
            Performs the mixup operation on a batch of data.

            Args:
                x (torch.Tensor): Input features tensor
                y (torch.Tensor): Target labels tensor
                z (Any, optional): Additional data to pass through unchanged. Defaults to None.

            Returns:
                tuple: A tuple containing:
                    - Mixed input features (torch.Tensor)
                    - Mixed target labels (torch.Tensor)
                    - Additional data z if provided (Any)

    Example:
        >>> mixup = Mixup(mix_beta=0.2)
        >>> mixed_x, mixed_y = mixup(features, labels)
    """

    def __init__(self, mix_beta, mixadd=False):
        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, x, y, z=None):
        bs = x.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)
        x_coeffs = coeffs.view((-1,) + (1,) * (x.ndim - 1))
        y_coeffs = coeffs.view((-1,) + (1,) * (y.ndim - 1))

        x = x_coeffs * x + (1 - x_coeffs) * x[perm]

        if self.mixadd:
            y = (y + y[perm]).clip(0, 1)
        else:
            y = y_coeffs * y + (1 - y_coeffs) * y[perm]

        if z:
            return x, y, z

        return x, y
