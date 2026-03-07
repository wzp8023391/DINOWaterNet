# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class SegFormerMLP(nn.Module):
    """
    Individual MLP layer for SegFormer head.
    """
    def __init__(self, input_dim, embed_dim):
        """
        Args:
            input_dim (int): Number of input channels.
            embed_dim (int): Number of output channels (embedding dimension).
        """
        super().__init__()
        # 1x1 convolution to project input channels to the unified embed_dim
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C_i, H_i, W_i)
        return self.proj(x)


class SegmentationHead(nn.Module):
    """
    Segmentation Head (MLP Decoder Head).
    It receives multi-scale features from four different stages of the encoder.
    """
    def __init__(self, in_channels, embed_dim, num_classes):
        """
        Args:
            in_channels (list[int]): Channels from the four encoder stages, e.g., [64, 128, 320, 512].
            embed_dim (int): Unified hidden dimension for fusion, e.g., 256 or 768.
            num_classes (int): Number of target classes for segmentation.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Initialize four MLP blocks to process features from each scale
        self.mlps = nn.ModuleList([
            SegFormerMLP(c, embed_dim) for c in in_channels
        ])

        # 2. Final fusion module
        # Processes concatenated features, typically using a 3x3 conv + BN + ReLU
        self.fuse = nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # 3. Final classifier (1x1 convolution)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): Four feature maps from the encoder, ordered from largest to smallest scale.
                                     e.g., [f1, f2, f3, f4] where f1 is (H/4, W/4) and f4 is (H/32, W/32).
        """
        # Determine the target size (usually the largest feature map size: H/4 x W/4)
        base_size = features[0].shape[2:] 
        
        # 1. Apply MLP to each feature map and upscale to the base size
        outs = []
        for i, (f, mlp) in enumerate(zip(features, self.mlps)):
            # 1.1 Project channels to embed_dim
            f = mlp(f) # (B, C_i, H_i, W_i) -> (B, embed_dim, H_i, W_i)
            
            # 1.2 Interpolate all feature maps to base_size (H/4, W/4)
            if f.shape[2:] != base_size:
                f = F.interpolate(f, 
                                 size=base_size, 
                                 mode='bilinear', 
                                 align_corners=False)
            outs.append(f)

        # 2. Concatenate features along the channel dimension
        fused = torch.cat(outs, dim=1) # (B, embed_dim * 4, H/4, W/4)
        
        # 3. Process the fused feature map
        fused = self.fuse(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        # 4. Final classification layer
        output = self.classifier(fused) # (B, num_classes, H/4, W/4)
        
        return output
 
 
# --- Basic Utility Functions ---
def window_partition(x, window_size):
    """
    Partitions the input tensor into non-overlapping windows.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Merges windows back into the original feature map.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

class Conv2d_BN(torch.nn.Sequential):
    """
    A sequential block consisting of Conv2d followed by BatchNorm2d.
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class Residual(torch.nn.Module):
    """
    Implements a residual connection with optional DropPath and LayerScale.
    """
    def __init__(self, m, drop_path=0., layer_scale_init_value=0, dim=None):
        super().__init__()
        self.m = m
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        if self.gamma is not None:
            return x + self.gamma * self.drop_path(self.m(x))
        else:
            return x + self.drop_path(self.m(x))


class SHMA(nn.Module):
    """
    Simplified Hierarchical Multi-head Attention (SHMA).
    """
    def __init__(self, dim, num_heads=1, attn_drop=0., ratio=4, q_kernel=1, kv_kernel=1,
                 head_dim_reduce_ratio=4, window_size=0, **kwargs):
        super().__init__()
        mid_dim = int(dim * ratio)
        dim_attn = dim // head_dim_reduce_ratio
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.v_dim_head = mid_dim // self.num_heads
        self.scale = self.dim_head ** -0.5
        self.window_size = window_size

        self.q = Conv2d_BN(dim, dim_attn, q_kernel, stride=1, pad=q_kernel // 2)
        self.k = Conv2d_BN(dim, dim_attn, kv_kernel, stride=1, pad=kv_kernel // 2)
        self.gate_act = nn.Sigmoid()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv2d_BN(mid_dim, dim, 1)
        self.v_gate = Conv2d_BN(dim, 2 * mid_dim, kv_kernel, stride=1, pad=kv_kernel // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        do_window = self.window_size > 0
        
        if do_window:
            # Padding to ensure dimensions are divisible by window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            Ho, Wo = H, W
            _, _, Hp, Wp = x.shape
            x = window_partition(x, self.window_size)
            curr_B, curr_C, curr_H, curr_W = x.shape
        else:
            curr_B, curr_C, curr_H, curr_W = B, C, H, W

        # Generate value and gate using a single conv layer then chunking
        v, gate = self.gate_act(self.v_gate(x)).chunk(2, dim=1)
        q = self.q(x).flatten(2)
        k = self.k(x).flatten(2)
        v = v.flatten(2)

        q = q * self.scale
        attn = q.transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (v @ attn.transpose(-2, -1)).view(curr_B, -1, curr_H, curr_W)
        x = x * gate
        x = self.proj(x)

        if do_window:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :Ho, :Wo].contiguous()
        return x

class SHMABlock(nn.Module):
    """
    Wrapper for SHMA that handles optional window splitting and reversing.
    """
    def __init__(self, window_split=False, window_reverse=False, drop_path=0.,
                 layer_scale_init_value=1e-6, **kargs):
        super().__init__()
        self.window_split = window_split
        self.window_reverse_flag = window_reverse
        self.window_size = kargs.get('window_size', 0)
        
        shma_args = kargs.copy()
        if window_split or window_reverse:
            shma_args['window_size'] = 0
        
        self.token_channel_mixer = Residual(
            SHMA(**shma_args),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=kargs['dim'],
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        info = (H, W, H, W, 0, 0)
        
        if self.window_split and ws > 0:
            pad_r = (ws - W % ws) % ws
            pad_b = (ws - H % ws) % ws
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            Hp, Wp = x.shape[2], x.shape[3]
            # Chunking channels to optimize window partitioning if necessary
            x_split = x.chunk(min(16, x.shape[1]), dim=1) 
            new_x = [window_partition(split, ws) for split in x_split]
            x = torch.cat(new_x, dim=1)
            info = (H, W, Hp, Wp, pad_r, pad_b)

        x = self.token_channel_mixer(x)

        if self.window_reverse_flag and ws > 0:
            Ho, Wo, Hp, Wp, pad_r, pad_b = info
            x_split = x.chunk(min(16, x.shape[1]), dim=1)
            new_x = [window_reverse(split, ws, Hp, Wp) for split in x_split]
            x = torch.cat(new_x, dim=1)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :Ho, :Wo].contiguous()
        return x

# --- Other Functional Blocks ---
class FFN2d(nn.Module):
    """
    2D Feed-Forward Network with 1x1 convolutions.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, ratio=4, **kargs):
        super().__init__()
        mid_chs = int(ratio * dim)
        self.channel_mixer = Residual(nn.Sequential(
            Conv2d_BN(dim, mid_chs), nn.GELU(), Conv2d_BN(mid_chs, dim)),
            drop_path=drop_path, layer_scale_init_value=layer_scale_init_value, dim=dim)
    def forward(self, x): return self.channel_mixer(x)

class ConvBlock(nn.Module):
    """
    Convolutional block with Depthwise and Pointwise operations.
    """
    def __init__(self, dim, out_dim=None, drop_path=0., layer_scale_init_value=1e-6,
                 kernel=7, stride=1, ratio=4, **kargs):
        super().__init__()
        mid_chs = int(ratio * dim)
        out_dim = out_dim or dim
        self.token_channel_mixer = Residual(nn.Sequential(
            Conv2d_BN(dim, dim, kernel, stride, pad=kernel // 2, groups=dim),
            Conv2d_BN(dim, mid_chs), nn.GELU(), Conv2d_BN(mid_chs, out_dim)),
            drop_path=drop_path, layer_scale_init_value=layer_scale_init_value, dim=out_dim)
    def forward(self, x): return self.token_channel_mixer(x)

class RepCPE(nn.Module):
    """
    Conditional Positional Encoding via depthwise convolution.
    """
    def __init__(self, dim, kernel=3, **kargs):
        super().__init__()
        self.cpe = Residual(Conv2d_BN(dim, dim, kernel, 1, pad=kernel//2, groups=dim))
    def forward(self, x): return self.cpe(x)

class BasicBlock(nn.Module):
    """
    Generic block dispatcher that instantiates specific blocks based on block_type string.
    """
    def __init__(self, dim, out_dim=None, drop_path=0., layer_scale_init_value=1e-6,
                 block_type=None, block_index=-1):
        super().__init__()
        args_dict = {
            "dim": dim, "out_dim": out_dim, "drop_path": drop_path,
            "layer_scale_init_value": layer_scale_init_value, "block_index": block_index,
            "window_size": 0
        }
        type_parts = block_type.split('_')
        block_name = type_parts[0]
        
        for arg in type_parts[1:]:
            key = ''.join(filter(str.isalpha, arg))
            val_str = ''.join(filter(lambda x: x.isdigit() or x=='.', arg))
            if not val_str: continue
            if key == 'k': args_dict['kernel'] = int(val_str)
            elif key == 'qk': args_dict['q_kernel'] = int(val_str)
            elif key == 'kvk': args_dict['kv_kernel'] = int(val_str)
            elif key == 'r': args_dict['ratio'] = float(val_str)
            elif key == 'hdrr': args_dict['head_dim_reduce_ratio'] = int(val_str)
            elif key == 'nh': args_dict['num_heads'] = int(val_str)
            elif key == 'ws': args_dict['window_size'] = int(val_str)
            elif key == 'wsp': args_dict['window_split'] = bool(int(val_str))
            elif key == 'wre': args_dict['window_reverse'] = bool(int(val_str))

        # Special handling: Default ws=8 for SHMA if not specified to avoid errors
        if 'SHMA' in block_name and args_dict['window_size'] == 0:
            args_dict['window_size'] = 8

        self.block = eval(block_name)(**args_dict)
    def forward(self, x): return self.block(x)

class EdgeResidual(nn.Module):
    """
    Residual block designed for edge/shallow features.
    """
    def __init__(self, in_chs, out_chs, exp_kernel_size=3, stride=1, exp_ratio=1.0):
        super().__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.conv_exp_bn1 = Conv2d_BN(in_chs, mid_chs, exp_kernel_size, stride, pad=exp_kernel_size//2)
        self.act = nn.ReLU()
        self.conv_pwl_bn2 = Conv2d_BN(mid_chs, out_chs, 1)
    def forward(self, x):
        return self.conv_pwl_bn2(self.act(self.conv_exp_bn1(x)))

# --- Main Model Class ---
class iFormer(nn.Module):
    def __init__(self, in_channels=3, depths=[2, 2, 16, 6], dims=[32, 64, 128, 256], numClass=2,
                 drop_path_rate=0., layer_scale_init_value=0,
                 block_types=None, downsample_kernels=[5, 3, 3, 3]):
        super().__init__()
        # Initial stem downsampling
        self.stem = nn.Sequential(
            Conv2d_BN(in_channels, dims[0] // 2, downsample_kernels[0], 2, downsample_kernels[0]//2),
            nn.GELU(),
            EdgeResidual(dims[0] // 2, dims[0], downsample_kernels[0], 2, exp_ratio=4)
        )

        self.downsamplers = nn.ModuleList([
            Conv2d_BN(dims[i], dims[i+1], downsample_kernels[i+1], 2, downsample_kernels[i+1]//2)
            for i in range(3)
        ])
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                BasicBlock(dim=dims[i], drop_path=dp_rates[cur+j],
                          layer_scale_init_value=layer_scale_init_value,
                          block_type=block_types[cur+j], block_index=cur+j)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
        self.EMBED_DIM = dims[2]  

        self.seg_head = SegmentationHead(
            in_channels=dims,
            embed_dim=self.EMBED_DIM,
            num_classes=numClass
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _ ,H, W = x.shape
        
        outs = []
        # Stage 1
        x = self.stages[0](self.stem(x))
        outs.append(x)
        # Stages 2, 3, 4
        for i in range(3):
            x = self.stages[i+1](self.downsamplers[i](x))
            outs.append(x)
            
        # Segmentation Head
        x = self.seg_head(outs)
        
        # Final upsampling to match input resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x


def iFormer_s(**kwargs):
    depths = [2, 2, 19, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 9 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_nh1_ws8', 'FFN2d_r3'] * 3 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_nh1_ws8', 'FFN2d_r3'] * 2
    return iFormer(depths=depths, dims=[32, 64, 176, 320], block_types=block_types, **kwargs)
