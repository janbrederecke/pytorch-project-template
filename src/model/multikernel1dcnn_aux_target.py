import torch
import torch.nn as nn


class ResNet_1D_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        downsampling,
        dropout_rate,
    ):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.LeakyReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fixed_kernel_size = 3
        self.kernels = config.KERNELS
        self.planes = config.PLANES
        self.parallel_conv = nn.ModuleList()
        self.in_channels = len(config.FEATURES)
        self.dropout_rate = config.DROPOUT_RATE
        self.training = config.TRAINING
        self.calculate_loss = torch.nn.CrossEntropyLoss()
        self.calculate_aux_loss = torch.nn.BCEWithLogitsLoss()

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=self.fixed_kernel_size,
            stride=2,
            padding=2,
            bias=False,
        )
        self.block = self._make_resnet_layer(
            kernel_size=self.fixed_kernel_size, stride=1, padding=self.fixed_kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.fc = nn.Linear(in_features=self.planes * 1, out_features=config.N_CLASSES)

        # Additional head for target1 (binary classification)
        self.fc_target1 = nn.Linear(in_features=self.planes * 1, out_features=1)  # Output 1 for binary classification

    def _make_resnet_layer(self, kernel_size, stride, blocks=6, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dropout_rate=self.dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, batch):
        x = batch["input"]
        if "target" in batch.keys():
            y = batch["target"]
        if "target1" in batch.keys():
            y1 = batch["target1"]

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avgpool(out)
        features = out.reshape(out.shape[0], -1)

        out_main = self.fc(features)

        # Pass features through the target1 classification head
        out_aux = self.fc_target1(features).squeeze(
            1
        )  # Squeeze to remove the last dimension of size 1 for BCEWithLogitsLoss

        outputs = {}

        # Calculate loss if target is available (training, validation)
        if "target" in batch.keys():
            loss_main = torch.stack(
                [self.calculate_loss(out_main[i], y.long()[i]) for i in range(len(out_main))]
            ).mean()
        if "target1" in batch.keys():
            loss_aux = torch.stack([self.calculate_aux_loss(out_aux[i], y1[i]) for i in range(len(out_aux))]).mean()
        outputs["loss"] = 0.9 * loss_main + 0.1 * loss_aux
        if not self.training:
            outputs["logits"] = out_main
        return outputs
