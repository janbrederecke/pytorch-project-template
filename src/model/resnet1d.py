import numpy as np
import torch
import torch.nn as nn

from src.model.gpu_mixup import Mixup


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """
    Residual network unit for unidimensional signals.
    """

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("This implementation only supports odd `kernel_size`.")
        super(ResBlock1d, self).__init__()

        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []

        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]

        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]

        # Build skip connection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """
        Residual unit.
        """

        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y

        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)

        # Sum skip connection and main connection
        x += y

        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """
    Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple should be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in,
            n_filters_out,
            kernel_size,
            bias=False,
            stride=downsample,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module("resblock1d_{0}".format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """
        Implement ResNet1d forward propagation
        """

        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.mean(-1)

        logits = self.lin(x)
        return logits


class Net(nn.Module):
    """ResNet-based neural network for time series classification.

    This class implements a neural network architecture based on ResNet1D for time series data,
    with optional mixup augmentation during training.

    Args:
        config: Configuration object containing the following attributes:
            - TRAINING (bool): Whether the model is in training mode
            - INPUT_DIM (int): Input dimension size
            - BLOCKS_DIM (list): List of block dimensions for ResNet
            - N_CLASSES (int): Number of output classes
            - KERNEL_SIZE (int): Kernel size for convolutions
            - DROPOUT_RATE (float): Dropout rate
            - MIX_BETA (float): Beta parameter for mixup
            - MIX_ADD (bool): Whether to use additive mixup
            - MIXUP (bool): Whether to use mixup augmentation
            - MIXUP_P (float): Probability of applying mixup

    Attributes:
        backbone (ResNet1d): The main ResNet1D architecture
        calculate_loss (nn.BCEWithLogitsLoss): Binary cross entropy loss function
        mixup (Mixup): Mixup augmentation module

    Methods:
        forward(batch): Performs forward pass through the network
            Args:
                batch (dict): Input batch containing:
                    - input (torch.Tensor): Input time series data
                    - target (torch.Tensor, optional): Target labels
                    - location (Any, optional): Location information

            Returns:
                dict: Dictionary containing:
                    - loss (torch.Tensor, optional): Computed loss if targets provided
                    - logits (torch.Tensor, optional): Model predictions in evaluation mode
                    - location (Any, optional): Location information in evaluation mode
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training = config.TRAINING
        self.backbone = ResNet1d(
            input_dim=config.INPUT_DIM,
            blocks_dim=config.BLOCKS_DIM,
            n_classes=config.N_CLASSES,
            kernel_size=config.KERNEL_SIZE,
            dropout_rate=config.DROPOUT_RATE,
        )
        self.calculate_loss = torch.nn.CrossEntropyLoss()
        self.mixup = Mixup(mix_beta=config.MIX_BETA, mixadd=config.MIX_ADD)

    def forward(self, batch):
        # Forward pass
        x = batch["input"]
        if "target" in batch.keys():
            y = batch["target"]
        # if self.training and self.config.MIXUP:
        #     if torch.rand(1)[0] < self.config.MIXUP_P:
        #         x, y = self.mixup(x, y)
        out = self.backbone(x)

        outputs = {}

        # Calculate loss if target is available (training, validation)
        if "target" in batch.keys():
            outputs["loss"] = torch.stack([self.calculate_loss(out[i], y.long()[i]) for i in range(len(out))]).mean()
        if not self.training:
            outputs["logits"] = out
        return outputs
