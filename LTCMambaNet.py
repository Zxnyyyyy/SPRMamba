from __future__ import annotations
import torch.nn as nn
import torch
from torch import Tensor
from mamba_ssm import Mamba, Mamba2
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import Optional, Union, Tuple
from einops import repeat, rearrange
import numpy as np

def exists(val):
    return val is not None

def patchify(ts: Tensor, patch_size: int, stride: int):
    patches = ts.unfold(2, patch_size, stride)  # [bs, d, nw, patch_size]
    return patches

def convert_to_patches(x: Tensor, patch_size: int, overlapping_patches: bool = False):
    _, _, seq_len = x.shape
    if seq_len % patch_size != 0:
        pad_size = patch_size - (seq_len % patch_size)
    else:
        pad_size = 0
    x = pad_sequence(x, pad_size)

    if overlapping_patches:
        half_pad = (patch_size//2, patch_size//2)
        padded_x = pad_sequence(x, pad_size=half_pad)
        patches = patchify(padded_x, 2*patch_size, stride=patch_size)
    else:
        patches = patchify(x, patch_size, stride=patch_size)
    return patches

def pad_sequence(x: Tensor, pad_size: Union[int, Tuple[int, int]]):
    bs, _, seq_len = x.shape

    if isinstance(pad_size, int):
        pad_size = (0, pad_size)

    if pad_size[-1] <= 0:
        return x
    padded_x = F.pad(x, pad=pad_size, value=0.0)
    return padded_x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_q]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj = nn.Sequential(nn.GELU(), nn.Conv1d(n_heads * d_model, d_model, kernel_size=1, bias=True))

        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_model, n_heads)


    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        if input_K is None:
            input_K = input_Q
        if input_V is None:
            input_V = input_Q

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_model]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_model]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_model]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(2, 3).reshape(batch_size, self.n_heads * self.d_model, -1)  # context: [batch_size, n_heads * d_model, len_q]
        output = self.dropout(self.proj(context)) # [batch_size, d_model, len_q]
        output = output.transpose(1, 2) # [batch_size, len_q, d_model]
        return output, attn

class DilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilated_conv = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.dilated_conv(x))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowedMamba(nn.Module):
    def __init__(self, windowed_mamba_w: int, model_dim: int, mamba_type: str, numhead: int):
        super().__init__()
        self.windowed_mamba_w = windowed_mamba_w

        if mamba_type == 'mamba_v1':
            self.mamba = Mamba(
                d_model=model_dim,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                bimamba_type='v1'
            )
        elif mamba_type == 'mamba_v2':
            self.mamba = Mamba(
                d_model=model_dim,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                bimamba_type='v2'
            )
        elif mamba_type == 'mamba2':
            self.mamba = Mamba2(
                d_model=model_dim,  # Model dimension d_model
                d_state=64,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                headdim=4,
            )
        self.attn = MultiHeadAttention(model_dim, numhead)
        self.mlp = Mlp(in_features=model_dim, hidden_features=model_dim * 4, act_layer=nn.GELU, drop=0.1)
    def _reshape(self, x: Tensor, overlapping_patches: bool):
        patches = convert_to_patches(x, self.windowed_mamba_w, overlapping_patches)
        patches = rearrange(patches, "b d num_patches patch_size -> (b num_patches) d patch_size")
        return patches.contiguous()

    def _undo_reshape(self, patches: Tensor, batch_size: int, orig_seq_len: int):
        num_patches = patches.shape[0] // batch_size
        x = rearrange(patches,
                      "(b num_patches) d patch_size -> b d (num_patches patch_size)",
                      num_patches=num_patches,
                      b=batch_size)

        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def forward(self, x: Tensor, q: Tensor):
        batch_size, _, seq_len = x.shape
        x = self._reshape(x, overlapping_patches=False)
        x = x.transpose(1, 2)
        x_mamba, x_sa = torch.chunk(x, chunks=2, dim=2)
        if q is None:
            q = x_sa
        else:
            q = self._reshape(q, overlapping_patches=False).transpose(1, 2)
        out_mamba = self.mamba(x_mamba)
        out_mamba = self.mlp(out_mamba)
        out_sa, attn = self.attn(q, x_sa, x_sa)
        out_sa = self.mlp(out_sa)
        out = torch.cat([out_mamba, out_sa], dim=2)
        out = out.transpose(1, 2)
        out = self._undo_reshape(out, batch_size, seq_len)
        return out

class LTContextMamba(nn.Module):
    def __init__(self, long_term_mamba_g, model_dim, mamba_type, numhead):
        super().__init__()
        self.long_term_mamba_g = long_term_mamba_g

        if mamba_type == 'mamba_v1':
            self.mamba = Mamba(
                d_model=model_dim,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                bimamba_type='v1'
            )
        elif mamba_type == 'mamba_v2':
            self.mamba = Mamba(
                d_model=model_dim,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                bimamba_type='v2'
            )
        elif mamba_type == 'mamba2':
            self.mamba = Mamba2(
                d_model=model_dim,  # Model dimension d_model
                d_state=64,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
                headdim=4,
            )
        self.attn = MultiHeadAttention(model_dim, numhead)
        self.mlp = Mlp(in_features=model_dim, hidden_features=model_dim * 4, act_layer=nn.GELU, drop=0.1)
    def _reshape(self, x: Tensor):
        patches = convert_to_patches(x, self.long_term_mamba_g)
        lt_patches = rearrange(patches, "b d num_patches patch_size -> (b patch_size) d num_patches")
        return lt_patches.contiguous()

    def _undo_reshape(self, lt_patches, batch_size, orig_seq_len):
        x = rearrange(lt_patches,
                      "(b patch_size) d num_patches -> b d (num_patches patch_size)",
                      patch_size=self.long_term_mamba_g,
                      b=batch_size)
        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def forward(self, x: Tensor, q: Tensor):
        batch_size, _, seq_len = x.shape # (B, C, L)
        x = self._reshape(x) # (B, C, L)
        x = x.transpose(1, 2) # (B, L, C)
        x_mamba, x_sa = torch.chunk(x, chunks=2, dim=2) # (B, L, C/2)
        if q is None:
            q = x_sa
        else:
            q = self._reshape(q).transpose(1, 2)
        out_mamba = self.mamba(x_mamba)
        out_mamba = self.mlp(out_mamba)
        out_sa, attn = self.attn(q, x_sa, x_sa) # (B, L, C/2)
        out_sa = self.mlp(out_sa)
        out = torch.cat([out_mamba, out_sa], dim=2) # (B, L, C)
        out = out.transpose(1, 2) # (B, C, L)
        out = self._undo_reshape(out, batch_size, seq_len) # (B, C, L)
        return out

class LTCMambaLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 windowed_mamba_w: int,
                 long_term_mamba_g: int,
                 dilation: int,
                 use_instance_norm: bool,
                 mamba_type: str,
                 drop_path=0.,
                 numhead=8):
        super().__init__()
        self.dilated_conv = DilatedConv(n_channels=dim, dilation=dilation, kernel_size=3)

        if use_instance_norm:
            self.instance_norm = nn.Identity()
        else:
            self.instance_norm = nn.InstanceNorm1d(dim)
        self.windowed_mamba = WindowedMamba(windowed_mamba_w=windowed_mamba_w, model_dim=dim // 2, mamba_type=mamba_type, numhead=numhead)
        self.ltc_mamba = LTContextMamba(long_term_mamba_g=long_term_mamba_g, model_dim=dim // 2, mamba_type=mamba_type, numhead=numhead)
        self.out_linear = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor, q: Tensor = None): # x:(B, C, T)
        out = self.dilated_conv(x) # (B, C, T)
        out = out + self.windowed_mamba(self.instance_norm(out), q) # (B, C, T)
        out = out + self.ltc_mamba(self.instance_norm(out), q) # (B, C, T)
        out = self.out_linear(out)
        out = self.drop_path(out)
        out = out + x
        return out

class mamba_block(nn.Module):
    """
    Args:
        num_layers (tuple(int)): Number of blocks at each stage. Default: 9
        input_dim (int): Number of input sequence channels. Default: 2048
        model_dim (int): Feature dimension at each stage. Default: 64
        num_classes (int): Number of classes for classification head. Default: 8
        dropout_prob (float): Stochastic depth rate. Default: 0.
    """

    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 model_dim: int,
                 windowed_mamba_w: int,
                 long_term_mamba_g: int,
                 num_classes: int,
                 dilation_factor: int,
                 use_instance_norm: bool,
                 mamba_type: str,
                 dropout_prob=0.,
                 numhead=8):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1, bias=True)
        self.stages = nn.ModuleList([])
        for i in range(num_layers):
            self.stages.append(
                LTCMambaLayer(
                    dim=model_dim,
                    dilation=dilation_factor ** i,
                    windowed_mamba_w=windowed_mamba_w,
                    long_term_mamba_g=long_term_mamba_g,
                    use_instance_norm=use_instance_norm,
                    mamba_type=mamba_type,
                    drop_path=dropout_prob,
                    numhead=numhead
                ))
        self.out_proj = nn.Conv1d(model_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, input: Tensor, q: Tensor = None):
        feature = self.input_proj(input)
        for stage in self.stages:
            feature = stage(feature, q)
        mamba_feature, sa_feature = torch.chunk(feature, chunks=2, dim=1)
        out = self.out_proj(feature)
        return out, sa_feature

class Temporal_mamba(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.stage1 = mamba_block(num_layers=args.NUM_LAYERS,
                                input_dim=args.INPUT_DIM,
                                model_dim=64,
                                num_classes=args.num_classes,
                                dilation_factor=2,
                                windowed_mamba_w=args.WINDOWED_Mamba_W,
                                long_term_mamba_g=args.LONG_TERM_Mamba_G,
                                use_instance_norm=True,
                                mamba_type=args.mamba_type,
                                dropout_prob=0.2,
                                numhead=args.numhead
                                )

        reduced_dim = int(64 // 2.0)
        self.dim_reduction = nn.Conv1d(reduced_dim, reduced_dim // 2, kernel_size=1, bias=True)
        self.stages = nn.ModuleList([])
        for s in range(1, args.NUM_STAGES):
            self.stages.append(
                mamba_block(num_layers=args.NUM_LAYERS,
                          input_dim=args.num_classes,
                          model_dim=reduced_dim,
                          num_classes=args.num_classes,
                          dilation_factor=2,
                          windowed_mamba_w=args.WINDOWED_Mamba_W,
                          long_term_mamba_g=args.LONG_TERM_Mamba_G,
                          use_instance_norm=True,
                          mamba_type=args.mamba_type,
                          dropout_prob=0.2,
                          numhead=args.numhead
                          )
            )
    def forward(self, inputs: Tensor) -> Tensor:
        out, sa_feature = self.stage1(inputs)
        feature_list = [sa_feature]
        output_list = [out]
        sa_feature = self.dim_reduction(sa_feature)
        for stage in self.stages:
            out, sa_feature = stage(F.softmax(out, dim=1), sa_feature)
            output_list.append(out)
            feature_list.append(sa_feature)
        logits = torch.stack(output_list, dim=0)
        return logits