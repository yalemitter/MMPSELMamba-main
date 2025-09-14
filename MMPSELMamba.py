import math
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from timm.models.registry import register_model


def _make_divisible(v, divisor, min_value = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks = 1, stride = 1, pad = 0, dilation = 1,
                 groups = 1, bn_weight_init = 1, bias = False,
                 norm_cfg = dict(type = 'BN', requires_grad = True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias = bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class DepthwiseConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, dilation=1,norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.dwconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=3, stride=upscale_factor, padding=dilation, output_padding=upscale_factor-1,
            dilation=dilation,groups=in_channels, bias=False
        )
        bn_dw = build_norm_layer(norm_cfg, in_channels)[1]
        self.bn_dw = nn.Sequential(
            bn_dw,
            nn.ReLU(inplace=True)
        )

        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        bn_pw = build_norm_layer(norm_cfg, out_channels)[1]
        self.bn_pw = nn.Sequential(
            bn_pw,

        )

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn_dw(x)
        x = self.pwconv(x)
        x = self.bn_pw(x)
        return x


class PointwiseConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1,norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.pwconv_down = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0,dilation=dilation, bias=False)
        bn_pw_down = build_norm_layer(norm_cfg, out_channels)[1]
        self.bn_pw_down = nn.Sequential(
            bn_pw_down,
        )

    def forward(self, x):
        x = self.pwconv_down(x)
        x = self.bn_pw_down(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.ReLU, drop = 0.,
                 norm_cfg = dict(type = 'BN', requires_grad = True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg = norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias = True, groups = hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg = norm_cfg)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            activations = None,
            norm_cfg = dict(type = 'BN', requires_grad = True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks = 1, norm_cfg = norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks = ks, stride = stride, pad = ks // 2, groups = hidden_dim,
                      norm_cfg = norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks = 1, norm_cfg = norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            z=self.conv(x)
            return x+z
        else:
            return self.conv(x)


class StackedMV2Block(nn.Module):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel = 16,
            activation = nn.ReLU,
            norm_cfg = dict(type = 'BN', requires_grad = True),
            width_mult = 1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg = norm_cfg),
                activation()
            )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks = k, stride = s, expand_ratio = t,
                                     norm_cfg = norm_cfg,
                                     activations = activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x,indices):
        B, C, N = x.shape
        indices = F.interpolate(indices.float(), size = (N,), mode = 'linear', align_corners = False)
        x = x + F.interpolate(self.pos_embed, size = (N), mode = 'linear', align_corners = False)+indices

        return x

class mmse_mamba(nn.Module):
    def __init__(
            self,
            dim,
            activation1 = nn.ReLU6,
            d_state = 16,
            d_conv = 4,
            expand = 2,
            device = None,
            dtype = None,
            # bimamba_type = "none",
            layer_idx = None,
            if_devide_out = False,
            init_layer_scale = None,
            norm_cfg = dict(type = 'BN', requires_grad = True),

    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.layer_idx = layer_idx
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale

        self.pos_emb_row = SqueezeAxialPositionalEmbedding(self.d_model, 16)
        self.pos_emb_column = SqueezeAxialPositionalEmbedding(self.d_model, 16)

        self.forward_mamba = Mamba(d_model = self.d_model, d_state = self.d_state, d_conv = self.d_conv,
                                       expand = self.expand, layer_idx = self.layer_idx,
                                       if_devide_out = self.if_devide_out, init_layer_scale = self.init_layer_scale)

        self.backward_mamba = Mamba(d_model = self.d_model, d_state = self.d_state, d_conv = self.d_conv,
                                        expand = self.expand, layer_idx = self.layer_idx,
                                        if_devide_out = self.if_devide_out,
                                        init_layer_scale = self.init_layer_scale)

        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(2 * self.d_model, self.d_model)
        self.upsample=DepthwiseConvUpsample(self.d_model,self.d_model,norm_cfg = norm_cfg)
        self.downsample=PointwiseConvDownsample(self.d_model,self.d_model,norm_cfg = norm_cfg)
        self.act1 = activation1()
        self.sigmoid = h_sigmoid()
        self.bn = nn.GroupNorm(num_groups=32,num_channels=self.d_model)
    def forward(self, x):
        B, C, H, W = x.shape

        z = x  # B, C, H, W
        z = self.act1(self.upsample(z))

        max_column_values, max_column_indices = x.max(dim = 2, keepdim = False)  #  B, C, H, W-->B, C, W
        min_column_values, min_column_indices = x.min(dim = 2, keepdim = False)  #  B, C, H, W-->B, C, W

        column_max = self.pos_emb_column(max_column_values,max_column_indices).permute(0, 2, 1)  #  B, C, W-->B, W, C
        column_max=self.norm(column_max)
        column_min = self.pos_emb_column(min_column_values,min_column_indices).permute(0, 2, 1)  #  B, C, W-->B, W, C
        column_min = self.norm(column_min)

        max_row_values, max_row_indices = x.max(dim = 3, keepdim = False)  #  B, C, H, W-->B,C,H
        min_row_values, min_row_indices = x.min(dim = 3, keepdim = False)  #  B, C, H, W-->B,C,H

        row_max = self.pos_emb_column(max_row_values,max_row_indices).permute(0, 2, 1)  #  B,C,H-->B,H,C
        row_max = self.norm(row_max)
        row_min = self.pos_emb_column(min_row_values,min_row_indices).permute(0, 2, 1)  #  B,C,H-->B,H,C
        row_min = self.norm(row_min)

        result_max = torch.cat((column_max, row_max), dim=1)
        result_min = torch.cat((column_min, row_min), dim = 1)
        forward_max = self.forward_mamba(result_max)
        forward_max = self.norm(forward_max)
        forward_min = self.forward_mamba(result_min)
        forward_min = self.norm(forward_min)
        backward_max = self.backward_mamba(result_max.flip([1]))
        backward_max = self.norm(backward_max)
        backward_min = self.backward_mamba(result_min.flip([1]))
        backward_min = self.norm(backward_min)
        res_max = torch.cat((forward_max, backward_max.flip([1])), dim = -1)
        res_min = torch.cat((forward_min, backward_min.flip([1])), dim = -1)

        res_max = self.output_proj(res_max).unsqueeze(dim = 2)
        res_max = self.norm(res_max)

        res_min = self.output_proj(res_min).unsqueeze(dim = 1)
        res_min = self.norm(res_min)
        res=res_max.add(res_min)
        res = res.permute(0, 3, 1, 2)
        res = self.sigmoid(res) * z
        res=self.bn(res)
        res = self.downsample(res)
        res = self.bn(res)
        res=res +x
        res=self.bn(res)
        return res


class Block(nn.Module):

    def __init__(self, dim, drop = 0.,
                 drop_path = 0., act_layer = nn.ReLU, norm_cfg = dict(type = 'BN2d', requires_grad = True)):
        super().__init__()
        self.dim = dim
        self.attn = mmse_mamba(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features = dim, hidden_features = 2*self.dim, act_layer = act_layer, drop = drop,
                       norm_cfg = norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim,
                  drop = 0., drop_path = 0.,
                 norm_cfg = dict(type = 'BN2d', requires_grad = True),
                 act_layer = None):
        super().__init__()
        self.block_num = block_num

        self.MMSE_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.MMSE_blocks.append(Block(
                embedding_dim,

                drop = drop, drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg = norm_cfg,
                act_layer = act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.MMSE_blocks[i](x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace = inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class MMSEMamba(nn.Module):
    def __init__(self, cfgs,
                 channels,
                 emb_dims,
                 depths = [2, 2],
                 drop_path_rate = 0.,
                 norm_cfg = dict(type = 'BN', requires_grad = True),
                 act_layer = nn.ReLU6,
                 init_cfg = None,
                 num_classes = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        if self.init_cfg is not None:
            self.pretrained = self.init_cfg['checkpoint']

        for i in range(len(cfgs)):
            smb = StackedMV2Block(cfgs = cfgs[i], stem = True if i == 0 else False, inp_channel = channels[i],
                                  norm_cfg = norm_cfg)
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(depths)):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[i])]  # stochastic depth decay rule
            MMSEs = BasicLayer(
                block_num = depths[i],
                embedding_dim = emb_dims[i],
                drop = 0,
                drop_path = dpr,
                norm_cfg = norm_cfg,
                act_layer = act_layer)
            setattr(self, f"MMSEs{i + 1}", MMSEs)

        self.linear = nn.Linear(channels[-1], self.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        num_smb_stage = len(self.cfgs)
        num_MMSEs_stage = len(self.depths)
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            if num_MMSEs_stage + i >= num_smb_stage:
                MMSEs = getattr(self, f"MMSEs{i + num_MMSEs_stage - num_smb_stage + 1}")
                x = MMSEs(x)
        out = self.avgpool(x).view(-1, x.shape[1])
        out = self.linear(out)

        return out

@register_model
def MMSEMamba_B(pretrained = False, **kwargs):
    model_cfgs = dict(
        cfg1 = [
            # k,  t,  c, s
            [3, 1, 16, 1],
            [3, 4, 32, 2],
            [3, 3, 32, 1]],
        cfg2 = [
            [5, 3, 64, 2],
            [5, 3, 64, 1]],
        cfg3 = [
            [3, 3, 128, 2],
            [3, 3, 128, 1]],
        cfg4 = [
            [5, 4, 192, 2]],
        cfg5 = [
            [3, 6, 192, 2]],
        channels = [16, 32, 64, 128, 192, 192],
        depths = [4, 4],
        emb_dims = [192, 192],
        drop_path_rate = 0.1,

    )
    return MMSEMamba(
        cfgs = [model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
        channels = model_cfgs['channels'],
        emb_dims = model_cfgs['emb_dims'],
        depths = model_cfgs['depths'],
        drop_path_rate = model_cfgs['drop_path_rate'])


@register_model
def MMPSELMamba_L(pretrained = False, **kwargs):
    model_cfgs = dict(
        cfg1 = [
            # k,  t,  c, s
            [3, 3, 32, 1],
            [3, 4, 64, 2],
            [3, 4, 64, 1]],
        cfg2 = [
            [5, 4, 128, 2],
            [5, 4, 128, 1]],
        cfg3 = [
            [3, 4, 192, 2],
            [3, 4, 192, 1]],
        cfg4 = [
            [5, 4, 192, 2]],
        cfg5 = [
            [3, 6, 192, 2]],
        # channels = [32, 64, 128, 192, 256, 320],
        channels = [32, 64, 128, 192, 192, 192],
        depths = [3, 3, 3],
        emb_dims = [192, 192, 192],
        drop_path_rate = 0.1,

        )
    return MMSEMamba(
        cfgs = [model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
        channels = model_cfgs['channels'],
        emb_dims = model_cfgs['emb_dims'],
        depths = model_cfgs['depths'],
        drop_path_rate = model_cfgs['drop_path_rate'])


from ptflops import get_model_complexity_info

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMPSELMamba_L().to(device)
    # print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {total_params}')

    input_res = (3, 224, 224)
    input = torch.rand((1, 3, 224, 224)).to(device)
    model.eval()
    b=model(input)
    print("b:",b.shape)

    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    print(f"FLOPs: {macs}, Parameters: {params}")

