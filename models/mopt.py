import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv2d_out_sz(kernel, stride, padding, dilation, in_sz):
    hout = int(
        (in_sz[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
    )
    wout = int(
        (in_sz[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
    )
    return [hout, wout]


class _LinEncoder(nn.Module):
    def __init__(
        self, conv_channels, apply_batch_norm, apply_layer_norm, in_sz, out_sz=128
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4800, 32),
            nn.LayerNorm([32]),
            nn.GELU(),
        )
        self.out_sz = out_sz

    def count_params(self):
        n_params = 0
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    n_params += p.numel()
        return n_params

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x


class _ConvEncoder(nn.Module):
    def __init__(
        self, conv_channels, apply_batch_norm, apply_layer_norm, in_sz, out_sz=32
    ):
        # def __init__(self, in_features, out_features, conv_channels):
        super().__init__()
        conv_layers = []
        dims = conv_channels
        sz = in_sz

        conv_layers.append(nn.Conv2d(dims[0], dims[1], 3, stride=2))
        sz = get_conv2d_out_sz((3, 3), (2, 2), (0, 0), (1, 1), sz)
        if apply_batch_norm:
            conv_layers.append(nn.BatchNorm2d(dims[1]))
        elif apply_layer_norm:
            conv_layers.append(nn.LayerNorm([dims[1]] + sz))
        conv_layers.append(nn.GELU())

        for i in range(1, len(dims) - 1):
            conv_layers.append(nn.Conv2d(dims[i], dims[i + 1], 3, stride=1))
            sz = get_conv2d_out_sz((3, 3), (1, 1), (0, 0), (1, 1), sz)
            if apply_batch_norm:
                conv_layers.append(nn.BatchNorm2d(dims[i + 1]))
            elif apply_layer_norm:
                conv_layers.append(nn.LayerNorm([dims[i + 1]] + sz))
            conv_layers.append(nn.GELU())
        self.conv_block = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1] * sz[0] * sz[1], out_sz),
            nn.LayerNorm([out_sz]),
            nn.GELU(),
        )
        self.out_sz = out_sz

    def count_params(self):
        n_params = 0
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    n_params += p.numel()
        return n_params

    def forward(self, x):
        x = self.conv_block(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class _ConvDecoder(nn.Module):
    def __init__(self, conv_channels, apply_batch_norm, apply_layer_norm, in_sz):
        super().__init__()

        conv_layers = []
        dims = conv_channels
        sz = in_sz
        for i in range(len(dims) - 1):
            conv_layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            sz = [2 * s for s in sz]
            conv_layers.append(nn.Conv2d(dims[i], dims[i + 1], 3, padding="same"))
            if apply_layer_norm:
                conv_layers.append(nn.LayerNorm([dims[i + 1]] + sz))
            elif apply_batch_norm:
                conv_layers.append(nn.BatchNorm2d(dims[i + 1]))
            conv_layers.append(nn.GELU())
        self.conv_block = nn.Sequential(*conv_layers)

    def count_params(self):
        n_params = 0
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    n_params += p.numel()
        return n_params

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ConvDecoderLin(nn.Module):
    def __init__(
        self,
        in_features,
        conv_channels,
        out_sz,
        apply_batch_norm=False,
        apply_layer_norm=False,
    ):
        super().__init__()
        _proj_layers = [nn.Linear(in_features, in_features)]
        if apply_layer_norm:
            _proj_layers += [nn.LayerNorm((in_features)), nn.GELU()]
        elif apply_batch_norm:
            _proj_layers += [nn.BatchNorm2d(in_features), nn.GELU()]
        self.proj_block = nn.Sequential(*_proj_layers)
        dec_channels = [in_features] + conv_channels
        self.conv_block = _ConvDecoder(
            dec_channels, apply_batch_norm, apply_layer_norm, [4, 4]
        )
        self.final_conv = nn.Conv2d(conv_channels[-1], out_sz, 1)

    def count_params(self):
        n_params = 0
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    n_params += p.numel()
        return n_params

    def forward(self, x):
        x = self.proj_block(x)
        # x = x.reshape(-1, 16, 8, 8)
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 4)
        x = self.conv_block(x)
        x = self.final_conv(x)
        return x


class ConvED(nn.Module):
    def __init__(
        self,
        in_features,
        conv_channels,
        in_sz,
        out_sz,
        action_sz,
        apply_batch_norm=False,
        apply_layer_norm=False,
        encoder="conv",
    ):
        super().__init__()

        enc_channels = conv_channels[:3]
        self.latent_sz = conv_channels[3]
        self.action_sz = action_sz
        # assert len(conv_channels) == 9
        enc_channels = [in_features] + enc_channels
        if encoder == "conv":
            self.encoder = _ConvEncoder(
                enc_channels,
                apply_batch_norm,
                apply_layer_norm,
                in_sz,
                out_sz=self.latent_sz,
            )
        elif encoder == "lin":
            self.encoder = _LinEncoder(
                enc_channels,
                apply_batch_norm,
                apply_layer_norm,
                in_sz,
                out_sz=self.latent_sz,
            )
        self.do = nn.Dropout(p=0.3)

        # Lin block
        lin_layers = [nn.Linear(self.latent_sz + action_sz, conv_channels[3])]
        # (bs, 16, 8, 8)
        if apply_layer_norm:
            lin_layers += [nn.LayerNorm((conv_channels[3])), nn.GELU()]
        elif apply_batch_norm:
            lin_layers += [nn.BatchNorm2d(conv_channels[3]), nn.GELU()]

        self.lin_block = nn.Sequential(*lin_layers)

        # Decode
        self.decoder = _ConvDecoder(
            conv_channels[4:],
            apply_batch_norm,
            apply_layer_norm,
            [3, 3],
        )
        self.end_conv = nn.Conv2d(conv_channels[-1], out_sz, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_params(self):
        n_params = 0
        for m in self.modules():
            for p in m.parameters():
                if p.requires_grad:
                    n_params += p.numel()
        return n_params

    def forward(self, state, action=None):
        x1 = self.encoder(state)
        x1 = x1.flatten(start_dim=1)
        x1 = self.do(x1)
        if self.action_sz > 0:
            x2 = torch.cat((x1, action), dim=-1)
        else:
            x2 = x1
        x = self.lin_block(x2)
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        x = self.decoder(x)
        x = self.end_conv(x)

        return x, x1
