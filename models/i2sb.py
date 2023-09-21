from logging import Filter
from locale import str
from builtins import NotImplementedError
from flax.training import common_utils
from cmath import isfinite
from functools import partial
from typing import Any, Callable, Sequence, Dict, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import math
import numpy as np
from einops import rearrange

from utils import expand_to_broadcast, jprint, batch_mul
from .bridge import Decoder as TinyDecoder, ResidualBlock

# revised from https://github.com/NVlabs/I2SB


class TrainModule(nn.Module):
    ...


class CtxModule(nn.Module):
    ...


class EmbedModule(nn.Module):
    ...


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    # b, c, *spatial = y[0].shape
    b, *spatial, c = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += jnp.array([matmul_ops], dtype=jnp.double)


class Upsample(TrainModule):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None
    # networks
    conv: nn.Module = nn.Conv

    @nn.compact
    def __call__(self, x, **kwargs):
        out_channels = self.out_channels or self.channels
        if self.use_conv:
            _features = out_channels
            _kernels = (3,)*self.dims
            _paddings = (1,)*self.dims
            conv = self.conv(_features, _kernels, padding=_paddings)

        # assert x.shape[1] == self.channels
        assert x.shape[-1] == self.channels
        if self.dims == 3:
            # B, C, D, H, W = x.shape
            # x = jax.image.resize(x, [B, C, D, 2*H, 2*W], "nearest")
            B, D, H, W, C = x.shape
            assert H == W
            assert D != H and D != W
            x = jax.image.resize(x, [B, D, 2*H, 2*W, C], "nearest")
        elif self.dims == 2:
            # B, C, H, W = x.shape
            # x = jax.image.resize(x, [B, C, 2*H, 2*W], "nearest")
            B, H, W, C = x.shape
            assert H == W
            x = jax.image.resize(x, [B, 2*H, 2*W, C], "nearest")
        elif self.dims == 1:
            B, W, C = x.shape
            x = jax.image.resize(x, [B, 2*W, C], "nearest")
        if self.use_conv:
            x = conv(x)
        return x


class Downsample(TrainModule):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None
    # networks
    conv: nn.Module = nn.Conv
    avgpool: Callable = nn.avg_pool

    @nn.compact
    def __call__(self, x, **kwargs):
        out_channels = self.out_channels or self.channels
        stride = (2,)*self.dims if self.dims != 3 else (1, 2, 2)
        if self.use_conv:
            _features = out_channels
            _kernels = (3,)*self.dims
            _strides = stride
            _paddings = (1,)*self.dims
            op = self.conv(
                _features, _kernels, strides=_strides, padding=_paddings
            )
        else:
            assert self.channels == out_channels
            _strides = stride
            op = partial(self.avgpool, window_shape=_strides, strides=_strides)

        # assert x.shape[1] == self.channels
        assert x.shape[-1] == self.channels
        return op(x)


class ResBlock(EmbedModule, TrainModule):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    use_conv: bool = False
    use_scale_shift_norm: bool = False
    dims: int = 2
    up: bool = False
    down: bool = False
    small: bool = False
    # networks
    silu: Callable = nn.silu
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    conv: nn.Module = nn.Conv
    # zero_conv: nn.Module = partial(
    #     nn.Conv,
    #     kernel_init=jax.nn.initializers.zeros,
    #     bias_init=jax.nn.initializers.zeros,
    #     name="frozen")
    zero_conv: nn.Module = nn.Conv
    upsample: nn.Module = Upsample
    downsample: nn.Module = Downsample
    dense: nn.Module = nn.Dense
    drop: nn.Module = nn.Dropout

    @nn.compact
    def __call__(self, x, emb, training=False, **kwargs):
        out_channels = self.out_channels or self.channels

        _features = out_channels
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        in_layers = [
            # self.norm(group_size=self.channels),
            self.norm(),
            self.silu,
            self.conv(_features, _kernels, padding=_paddings),
        ]

        updown = self.up or self.down

        if self.up:
            h_upd = self.upsample(self.channels, False, self.dims)
            x_upd = self.upsample(self.channels, False, self.dims)
        elif self.down:
            h_upd = self.downsample(self.channels, False, self.dims)
            x_upd = self.downsample(self.channels, False, self.dims)
        else:
            h_upd = x_upd = lambda x: x

        emb_layers = nn.Sequential([
            self.silu,
            self.dense(
                2 * out_channels if self.use_scale_shift_norm else out_channels,
            ),
        ])
        _features = out_channels
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        out_layers = nn.Sequential([
            # self.norm(group_size=out_channels),
            self.norm(),
            self.silu,
            self.drop(self.dropout, deterministic=not training),
            self.zero_conv(_features, _kernels, padding=_paddings)
        ])

        if out_channels == self.channels:
            def skip_connection(x): return x
        elif self.use_conv:
            _features = out_channels
            _kernels = (3,)*self.dims
            _paddings = (1,)*self.dims
            skip_connection = self.conv(
                _features, _kernels, padding=_paddings
            )
        else:
            _features = out_channels
            _kernels = (1,)*self.dims
            skip_connection = self.conv(_features, _kernels)

        if not self.small:
            if updown:
                in_rest, in_conv = in_layers[:-1], in_layers[-1]
                in_rest = nn.Sequential(in_rest)
                h = in_rest(x)
                h = h_upd(h)
                x = x_upd(x)
                h = in_conv(h)
            else:
                in_layers = nn.Sequential(in_layers)
                h = in_layers(x)
        else:
            h = x
        # emb_out = jnp.asarray(emb_layers(emb), h.dtype)
        emb_out = emb_layers(emb)
        B, E = emb_out.shape
        # expand = len(h.shape) - len(emb_out.shape)
        # init_axis = len(emb_out.shape)
        # expand = list(range(init_axis, init_axis+expand))
        # emb_out = jnp.expand_dims(emb_out, axis=expand)
        expand = (1,)*self.dims
        emb_out = emb_out.reshape(B, *expand, E)
        if self.use_scale_shift_norm:
            out_norm, out_rest = out_layers[0], out_layers[1:]
            # scale, shift = jnp.split(emb_out, 2, axis=1)
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = out_layers(h)
        return skip_connection(x) + h


class TinyResBlock(nn.Module):
    channels: int
    dropout: float
    reduced: bool = False
    use_batchnorm: bool = False
    # networks
    silu: Callable = nn.silu
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    batchnorm: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                   scale_init=jax.nn.initializers.ones,
                                   bias_init=jax.nn.initializers.zeros)
    conv: nn.Module = nn.Conv
    dense: nn.Module = nn.Dense
    drop: nn.Module = nn.Dropout

    @nn.compact
    def __call__(self, x, emb, **kwargs):
        ch = self.channels
        residual = x
        if self.reduced:
            emb = self.dense(ch)(emb)
            emb = emb[:, None, None, :]
        else:
            emb = self.silu(emb)
            emb = self.dense(ch)(emb)
            emb = emb[:, None, None, :]
        x = x+emb
        if self.use_batchnorm:
            x = self.batchnorm(use_running_average=not kwargs["training"])(x)
        else:
            x = self.norm()(x)
        x = self.silu(x)
        if self.dropout > 0:
            x = self.drop(
                self.dropout,
                deterministic=not kwargs["training"]
            )(x)
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        x = x+residual
        return x


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    n_heads: int

    @nn.compact
    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        qkv = jnp.transpose(qkv, (0, 2, 1))
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.reshape(bs * self.n_heads, ch * 3, length)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        dtype = weight.dtype
        weight = jax.nn.softmax(weight, axis=-1)
        weight = jnp.asarray(weight, dtype=dtype)
        a = jnp.einsum("bts,bcs->bct", weight, v)
        # return a.reshape(bs, -1, length)
        a = a.reshape(bs, -1, length)
        return jnp.transpose(a, (0, 2, 1))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    n_heads: int

    @nn.compact
    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        qkv = jnp.transpose(qkv, (0, 2, 1))
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.n_heads, ch, length),
            (k * scale).reshape(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        dtype = weight.dtype
        weight = jax.nn.softmax(weight, axis=-1)
        weight = jnp.asarray(weight, dtype=dtype)
        a = jnp.einsum("bts,bcs->bct", weight,
                       v.reshape(bs * self.n_heads, ch, length))
        # return a.reshape(bs, -1, length)
        a = a.reshape(bs, -1, length)
        return jnp.transpose(a, (0, 2, 1))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(TrainModule):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_checkpoint: bool = False
    use_new_attention_order: bool = False
    # networks
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    conv: nn.Module = nn.Conv
    qkv_attn: nn.Module = QKVAttention
    qkv_attn_legacy: nn.Module = QKVAttentionLegacy
    # zero_conv: nn.Module = partial(
    #     nn.Conv,
    #     kernel_init=jax.nn.initializers.zeros,
    #     bias_init=jax.nn.initializers.zeros,
    #     name="frozen")
    zero_conv: nn.Module = nn.Conv

    @nn.compact
    def __call__(self, x, **kwargs):

        if self.num_head_channels == -1:
            num_heads = self.num_heads
        else:
            assert (
                self.channels % self.num_head_channels == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_head_channels {self.num_head_channels}"
            num_heads = self.channels // self.num_head_channels

        # norm = self.norm(groups_size=self.channels)
        norm = self.norm()
        _features = 3*self.channels
        _kernels = (1,)
        qkv_fn = self.conv(_features, _kernels)
        if self.use_new_attention_order:
            # split qkv before split heads
            attention = self.qkv_attn(num_heads)
        else:
            # split heads before split qkv
            attention = self.qkv_attn_legacy(num_heads)

        _features = self.channels
        _kernels = (1,)
        proj_out = self.zero_conv(_features, _kernels)

        # b, c, *spatial = x.shape
        b, *spatial, c = x.shape
        # x = x.reshape(b, c, -1)
        x = x.reshape(b, -1, c)
        qkv = qkv_fn(norm(x))
        h = attention(qkv)
        h = proj_out(h)
        # return (x + h).reshape(b, c, *spatial)
        return (x + h).reshape(b, *spatial, c)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = timesteps[..., None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if len(timesteps.shape) == 2:
        embedding = rearrange(embedding, "b n d -> b (n d)")
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class Sequential(nn.Sequential):
    def __call__(self, x, *args, **kwargs):
        return super().__call__(x)


class EmbedSequential(nn.Sequential, EmbedModule, TrainModule, CtxModule):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __call__(self, x, emb=None, training=False, ctx=None, **kwargs):
        # def __call__(self, *args, **kwargs):
        # for layer in self.layers:
        #     if emb is not None:
        #         x = layer(x, emb, **kwargs)
        #     else:
        #         x = layer(x, **kwargs)
        # return x
        if not self.layers:
            raise ValueError(f'Empty Sequential module {self.name}.')

        # for layer in self.layers:
        #     _kwargs = dict()
        #     if isinstance(layer, TrainModule):
        #         _kwargs["training"] = training
        #     if isinstance(layer, CtxModule):
        #         _kwargs["ctx"] = ctx

        #     if isinstance(layer, EmbedModule):
        #         _args = (x, emb)
        #     else:
        #         _args = (x,)
        #     x = layer(*_args, **_kwargs)
        for layer in self.layers:
            x = layer(x, emb=emb, training=training, ctx=ctx)
        return x


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """
    # parameters
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: Sequence
    num_res_blocks: int
    attention_resolutions: Sequence
    dropout: float = 0.
    channel_mult: Sequence = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    dtype: Any = jnp.float32
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    context: Sequence = None
    # networks
    silu: Callable = nn.silu
    dense: nn.Module = nn.Dense
    embed: nn.Module = nn.Embed
    conv: nn.Module = nn.Conv
    # zero_conv: nn.Module = partial(
    #     nn.Conv,
    #     kernel_init=jax.nn.initializers.zeros,
    #     bias_init=jax.nn.initializers.zeros,
    #     name="frozen")
    zero_conv: nn.Module = nn.Conv
    res_block: nn.Module = ResBlock
    att_block: nn.Module = AttentionBlock
    downsample: nn.Module = Downsample
    upsample: nn.Module = Upsample
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    embed_sequential: nn.Module = EmbedSequential
    sequential: nn.Module = Sequential

    @nn.compact
    def __call__(self, x, t, y=None, **kwargs):
        timesteps = t
        # if self.context is None:
        #     del kwargs["ctx"]

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        assert x.shape[1] == x.shape[2]
        assert len(x.shape) == 4
        # B, H, W, C -> B, C, H, W
        # x = jnp.transpose(x, (0, 3, 1, 2))
        # assert x.shape[2] == x.shape[3]

        if self.num_heads_upsample == -1:
            num_heads_upsample = self.num_heads
        else:
            num_heads_upsample = self.num_heads_upsample

        time_embed_dim = self.model_channels * 4
        time_embed = self.sequential([
            self.dense(time_embed_dim),
            self.silu,
            self.dense(time_embed_dim),
        ])

        if self.num_classes is not None:
            label_emb = self.embed(time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        _features = ch
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        input_blocks = [
            self.sequential([
                self.conv(_features, _kernels, padding=_paddings)
            ])
        ]

        _feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    self.res_block(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        self.att_block(
                            ch,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                input_blocks.append(self.embed_sequential(layers))
                _feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                input_blocks.append(
                    self.embed_sequential([
                        self.res_block(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                        )
                        if self.resblock_updown
                        else self.downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    ])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                _feature_size += ch

        middle_block = self.embed_sequential([
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            self.att_block(
                ch,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        ])
        _feature_size += ch

        output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    self.res_block(
                        ch + ich,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        self.att_block(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        self.res_block(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else self.upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                output_blocks.append(self.embed_sequential(layers))
                _feature_size += ch

        _features = self.out_channels
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        out_fn = self.sequential([
            # self.norm(group_size=ch),
            self.norm(),
            self.silu,
            self.zero_conv(_features, _kernels, padding=_paddings),
        ])

        # forward
        hs = []
        emb = time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + label_emb(y)

        h = jnp.asarray(x, self.dtype)
        for module in input_blocks:
            h = module(h, emb, **kwargs)
            hs.append(h)
        h = middle_block(h, emb, **kwargs)
        for module in output_blocks:
            # h = jnp.concatenate([h, hs.pop()], axis=1)
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb, **kwargs)
        h = jnp.asarray(h, x.dtype)
        h = out_fn(h)
        # # B, C, H, W -> B, H, W, C
        # h = jnp.transpose(h, (0, 2, 3, 1))
        assert h.shape[1] == h.shape[2]

        return h


class MidUNetModel(nn.Module):
    # parameters
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: Sequence
    num_res_blocks: int
    attention_resolutions: Sequence
    dropout: float = 0.
    channel_mult: Sequence = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    dtype: Any = jnp.float32
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    context: Sequence = None
    # networks
    silu: Callable = nn.silu
    dense: nn.Module = nn.Dense
    embed: nn.Module = nn.Embed
    conv: nn.Module = nn.Conv
    # zero_conv: nn.Module = partial(
    #     nn.Conv,
    #     kernel_init=jax.nn.initializers.zeros,
    #     bias_init=jax.nn.initializers.zeros,
    #     name="frozen")
    zero_conv: nn.Module = nn.Conv
    res_block: nn.Module = ResBlock
    att_block: nn.Module = AttentionBlock
    downsample: nn.Module = Downsample
    upsample: nn.Module = Upsample
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    embed_sequential: nn.Module = EmbedSequential
    sequential: nn.Module = Sequential

    @nn.compact
    def __call__(self, x, t, y=None, **kwargs):
        timesteps = t
        # if self.context is None:
        #     del kwargs["ctx"]

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        assert x.shape[1] == x.shape[2]
        assert len(x.shape) == 4
        # B, H, W, C -> B, C, H, W
        # x = jnp.transpose(x, (0, 3, 1, 2))
        # assert x.shape[2] == x.shape[3]

        if self.num_heads_upsample == -1:
            num_heads_upsample = self.num_heads  # 1
        else:
            num_heads_upsample = self.num_heads_upsample

        # ---------------------------------------------------------------------
        # time embedding
        # ---------------------------------------------------------------------
        time_embed_dim = self.model_channels  # 256
        time_embed = self.sequential([
            self.dense(time_embed_dim),
            self.silu,
            self.dense(time_embed_dim),
        ])

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)  # 256
        _features = ch
        _kernels = (3,)*self.dims  # (3,3)
        _paddings = (1,)*self.dims  # (1,1)
        input_blocks = [
            self.sequential([
                self.conv(_features, _kernels, padding=_paddings)
            ])
        ]

        _feature_size = ch  # 256
        input_block_chans = [ch]
        ds = 1
        layers = [
            self.res_block(
                ch,  # 256
                time_embed_dim,  # 256
                self.dropout,  # 0.2
                out_channels=int(self.model_channels),  # 256
                dims=self.dims,  # 2
                use_scale_shift_norm=self.use_scale_shift_norm,  # False
            )
        ]
        ch = int(self.model_channels)  # 256
        input_blocks.append(self.embed_sequential(layers))
        _feature_size += ch  # 512
        input_block_chans.append(ch)  # [256, 256]

        middle_block = self.embed_sequential([
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            # self.att_block(
            #     ch,
            #     num_heads=self.num_heads,
            #     num_head_channels=self.num_head_channels,
            #     use_new_attention_order=self.use_new_attention_order,
            # ),
            # self.res_block(
            #     ch,
            #     time_embed_dim,
            #     self.dropout,
            #     dims=self.dims,
            #     use_scale_shift_norm=self.use_scale_shift_norm,
            # ),
        ])
        _feature_size += ch  # 768

        output_blocks = []
        for i in range(self.num_res_blocks + 1):  # 2
            ich = input_block_chans.pop()  # 256 -> 256
            layers = [
                self.res_block(
                    ch + ich,  # 256+256 -> 256+256
                    # ch,
                    time_embed_dim,  # 256
                    self.dropout,
                    out_channels=int(self.model_channels),  # 256
                    dims=self.dims,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                )
            ]
            ch = int(self.model_channels)
            output_blocks.append(self.embed_sequential(layers))
            _feature_size += ch  # 1024 -> 1280

        _features = self.out_channels  # 128
        _kernels = (3,)*self.dims  # (3,3)
        _paddings = (1,)*self.dims  # (1,1)
        out_fn = self.sequential([
            # self.norm(group_size=ch),
            self.norm(),
            self.silu,
            self.zero_conv(_features, _kernels, padding=_paddings),
        ])

        # forward
        hs = []
        emb = time_embed(timestep_embedding(timesteps, self.model_channels))

        # h = jnp.asarray(x, self.dtype)
        h = x
        for module in input_blocks:
            h = module(h, emb, **kwargs)
            hs.append(h)
        h = middle_block(h, emb, **kwargs)
        for module in output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb, **kwargs)
        # h = jnp.asarray(h, x.dtype)
        h = out_fn(h)
        # # B, C, H, W -> B, H, W, C
        # h = jnp.transpose(h, (0, 2, 3, 1))
        assert h.shape[1] == h.shape[2]

        return h


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution.
    """
    features: int
    kernel_size: Tuple[int]
    strides: Tuple[int]
    padding: Tuple[int]
    use_bias: bool = True
    kernel_init: Any = jax.nn.initializers.he_normal()
    bias_init: Any = jax.nn.initializers.zeros
    kernels_per_layer: int = 1

    @nn.compact
    def __call__(self, x):
        Conv = partial(
            nn.Conv,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        B, H, W, C = x.shape
        x = Conv(
            features=C * self.kernels_per_layer,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            feature_group_count=C
        )(x)  # depthwise
        x = Conv(
            features=self.features,
            kernel_size=(1, 1),
            padding='SAME'
        )(x)  # pointwise
        return x


class TinyUNetModel(nn.Module):
    ver: str
    # parameters
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: Sequence
    num_res_blocks: int
    attention_resolutions: Sequence
    dropout: float = 0.
    channel_mult: Sequence = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    dtype: Any = jnp.float32
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    context: Sequence = None
    # networks
    silu: Callable = nn.silu
    dense: nn.Module = nn.Dense
    embed: nn.Module = nn.Embed
    conv: nn.Module = nn.Conv
    # zero_conv: nn.Module = partial(
    #     nn.Conv,
    #     kernel_init=jax.nn.initializers.zeros,
    #     bias_init=jax.nn.initializers.zeros,
    #     name="frozen")
    zero_conv: nn.Module = nn.Conv
    res_block: nn.Module = partial(ResBlock, small=True)
    att_block: nn.Module = AttentionBlock
    downsample: nn.Module = Downsample
    upsample: nn.Module = Upsample
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    batchnorm: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                   scale_init=jax.nn.initializers.ones,
                                   bias_init=jax.nn.initializers.zeros)
    embed_sequential: nn.Module = EmbedSequential
    sequential: nn.Module = Sequential
    drop: nn.Module = nn.Dropout
    tiny_resblock: nn.Module = TinyResBlock

    @nn.compact
    def __call__(self, *args, **kwargs):
        if self.ver == "v1.0":
            return self._call_v1_0(*args, **kwargs)
        elif self.ver == "v1.1":
            return self._call_v1_1(*args, **kwargs)
        elif self.ver == "v1.2":
            return self._call_v1_2(*args, **kwargs)
        elif self.ver == "v1.3":
            return self._call_v1_3(*args, **kwargs)
        elif self.ver == "v1.4":
            # Depthwise separable convolution applied, 1 layer.
            return self._call_v1_4(*args, **kwargs)
        elif self.ver == "v1.5":
            # Depthwise separable convolution applied, 2 layers.
            return self._call_v1_5(*args, **kwargs)
        else:
            raise NotImplementedError

    def _call_v1_0(self, x, t, y=None, **kwargs):
        timesteps = t
        # ---------------------------------------------------------------------
        # time embedding
        # ---------------------------------------------------------------------
        time_embed_dim = self.model_channels//4  # 256
        time_embed = self.sequential([
            self.dense(time_embed_dim),
            self.silu,
            self.dense(time_embed_dim),
        ])

        ch = self.model_channels//2
        _features = ch
        _kernels = (3,)*self.dims  # (3,3)
        _paddings = (1,)*self.dims  # (1,1)
        input_blocks = [
            self.sequential([
                self.conv(_features, _kernels, padding=_paddings)
            ])
        ]

        input_block_chans = [ch]
        # embed_sequential
        # for layer in self.layers:
        #   x = layer(x, emb=emb, training=training, ctx=ctx)
        middle_block = self.embed_sequential([
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        ])

        output_blocks = []
        ich = input_block_chans.pop()  # 256 -> 256
        layers = [
            self.res_block(
                ch + ich,  # 256+256 -> 256+256
                time_embed_dim,  # 256
                self.dropout,
                out_channels=int(self.model_channels),  # 256
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            )
        ]
        output_blocks.append(self.embed_sequential(layers))

        _features = self.out_channels  # 128
        _kernels = (3,)*self.dims  # (3,3)
        _paddings = (1,)*self.dims  # (1,1)
        out_fn = self.sequential([
            # self.norm(group_size=ch),
            self.norm(),
            self.silu,
            self.zero_conv(_features, _kernels, padding=_paddings),
        ])

        # forward
        hs = []
        emb = time_embed(timestep_embedding(timesteps, self.model_channels))

        # h = jnp.asarray(x, self.dtype)
        h = x
        for module in input_blocks:
            h = module(h, emb, **kwargs)
            hs.append(h)
        h = middle_block(h, emb, **kwargs)
        for module in output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb, **kwargs)
        # h = jnp.asarray(h, x.dtype)
        h = out_fn(h)
        # # B, C, H, W -> B, H, W, C
        # h = jnp.transpose(h, (0, 2, 3, 1))
        # assert h.shape[1] == h.shape[2]

        return h

    def _call_v1_1(self, x, t, **kwargs):
        out_ch = x.shape[-1]
        t = timestep_embedding(t, self.model_channels)
        t_dim = self.model_channels//4  # 32
        t = self.dense(t_dim)(t)
        t = self.silu(t)
        t = self.dense(t_dim)(t)

        ch = self.model_channels//2  # 64
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)

        ch = self.model_channels//2  # 64
        residual = x
        _t = self.silu(t)
        _t = self.dense(ch)(_t)
        _t = _t.reshape(-1, 1, 1, ch)
        x = x+_t
        x = self.norm()(x)
        x = self.silu(x)
        x = self.drop(
            self.dropout,
            deterministic=not kwargs["training"]
        )(x)
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        x = x + residual

        x = jnp.concatenate([x, residual], axis=-1)

        ch = self.model_channels
        residual = x
        _t = self.silu(t)
        _t = self.dense(ch)(_t)
        _t = _t.reshape(-1, 1, 1, ch)
        x = x+_t
        x = self.norm()(x)
        x = self.silu(x)
        x = self.drop(
            self.dropout,
            deterministic=not kwargs["training"]
        )(x)
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        x = x+residual

        x = self.norm()(x)
        x = self.silu(x)
        x = self.conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)

        return x

    def _call_v1_2(self, x, t, **kwargs):
        out_ch = x.shape[-1]
        t = timestep_embedding(t, self.model_channels)
        # t = jnp.tile(t.reshape(-1, 1), [1, self.model_channels])
        t_dim = self.model_channels//4  # 32
        t = self.dense(t_dim)(t)
        t = self.silu(t)
        t = self.dense(t_dim)(t)

        ch = self.model_channels//2  # 64
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        residual = x

        ch = self.model_channels//2  # 64
        x = self.tiny_resblock(
            ch, self.dropout
        )(x, t, **kwargs)

        x = jnp.concatenate([x, residual], axis=-1)

        ch = self.model_channels
        x = self.tiny_resblock(
            ch, self.dropout
        )(x, t, **kwargs)

        x = self.norm()(x)
        x = self.silu(x)
        x = self.conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        return x

    def _call_v1_3(self, x, t, **kwargs):
        out_ch = x.shape[-1]
        t = timestep_embedding(t, self.model_channels)
        t_dim = self.model_channels//2  # 64
        t = self.dense(t_dim)(t)
        t = self.silu(t)

        ch = self.model_channels//2  # 64
        x = self.conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        residual = x

        ch = self.model_channels//2  # 64
        x = self.tiny_resblock(
            ch, self.dropout, reduced=True,
            use_batchnorm=True
        )(x, t, **kwargs)

        x = jnp.concatenate([x, residual], axis=-1)

        ch = self.model_channels
        x = self.tiny_resblock(
            ch, self.dropout, reduced=True,
            use_batchnorm=True
        )(x, t, **kwargs)

        x = self.batchnorm(
            use_running_average=not kwargs["training"]
        )(x)
        x = self.silu(x)
        x = self.conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        return x

    def _call_v1_4(self, x, t, **kwargs):
        out_ch = x.shape[-1]
        conv = DepthwiseSeparableConv
        # time embedding
        t = timestep_embedding(t, self.model_channels)
        # t = jnp.tile(t.reshape(-1, 1), [1, self.model_channels])
        t_dim = self.model_channels//4  # 32
        t = self.dense(t_dim)(t)
        t = self.silu(t)
        t = self.dense(t_dim)(t)

        # Conv
        ch = self.model_channels//2  # 64
        x = conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        residual = x

        # Downsample
        ch = self.model_channels//2  # 64
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        x = jnp.concatenate([x, residual], axis=-1)

        # Upsample
        ch = self.model_channels
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        x = self.norm()(x)
        x = self.silu(x)
        x = conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        return x

    def _call_v1_5(self, x, t, **kwargs):
        out_ch = x.shape[-1]
        conv = DepthwiseSeparableConv
        # time embedding
        t = timestep_embedding(t, self.model_channels)
        # t = jnp.tile(t.reshape(-1, 1), [1, self.model_channels])
        t_dim = self.model_channels//4  # 32
        t = self.dense(t_dim)(t)
        t = self.silu(t)
        t = self.dense(t_dim)(t)

        # Conv
        ch = self.model_channels//2  # 64
        x = conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        residual = x

        # Downsample
        # ch = self.model_channels // 4  # 32
        ch = x.shape[-1]  # 64
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        ch = self.model_channels // 4  # 32
        x = conv(
            features=ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        residual2 = x

        # Downsample
        # ch = self.model_channels // 4  # 64
        ch = x.shape[-1]
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        # Upsample
        x = jnp.concatenate([x, residual2], axis=-1)
        # ch = self.model_channels // 2 # 64
        ch = x.shape[-1]
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        # Upsample
        x = jnp.concatenate([x, residual], axis=-1)
        # ch = self.model_channels # 64
        ch = x.shape[-1]
        x = self.tiny_resblock(
            ch, self.dropout, conv=conv
        )(x, t, **kwargs)

        x = self.norm()(x)
        x = self.silu(x)
        x = conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1)
        )(x)
        return x


def to_norm_kwargs(norm, kwargs):
    return dict(
        use_running_average=not kwargs["training"]
    ) if getattr(norm, "func", norm) is nn.BatchNorm else dict()


class Conv1x1(nn.Module):
    ch: int
    conv: nn.Module = nn.Conv
    norm: nn.Module = nn.BatchNorm
    relu6: Callable = nn.activation.relu6

    @nn.compact
    def __call__(self, x, **kwargs):
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        x = self.conv(
            features=self.ch,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=False
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu6(x)
        return x


class Conv3x3(nn.Module):
    ch: int
    st: int
    conv: nn.Module = nn.Conv
    norm: nn.Module = nn.BatchNorm
    relu6: Callable = nn.activation.relu6

    @nn.compact
    def __call__(self, x, **kwargs):
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(self.st, self.st),
            padding=1,
            use_bias=False
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu6(x)
        return x


class InvertedResidual(nn.Module):
    in_ch: int
    ch: int
    st: int
    expand: int
    conv: nn.Module = nn.Conv
    norm: nn.Module = nn.BatchNorm
    relu6: Callable = nn.activation.relu6
    fc: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, x, **kwargs):
        residual = x
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        hidden_dim = round(self.in_ch*self.expand)
        identity = self.st == 1 and self.in_ch == self.ch
        if self.expand == 1:
            _x = x
            x = self.conv(
                features=hidden_dim,
                kernel_size=(3, 3),
                strides=(self.st, self.st),
                padding=1,
                feature_group_count=hidden_dim,
                use_bias=False
            )(x)
            # print(f"{x.shape[1]*x.shape[2]*3**2*hidden_dim}")
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            _x = x
            x = self.conv(
                features=self.ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            # print(f"{x.shape[1]*x.shape[2]*_x.shape[-1]*x.shape[-1]}")
            x = self.norm(**norm_kwargs)(x)
        else:
            _x = x
            x = self.conv(
                features=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            # print(f"{x.shape[1]*x.shape[2]*_x.shape[-1]*x.shape[-1]}")
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            _x = x
            x = self.conv(
                features=hidden_dim,
                kernel_size=(3, 3),
                strides=(self.st, self.st),
                padding=1,
                feature_group_count=hidden_dim,
                use_bias=False
            )(x)
            # print(f"{x.shape[1]*x.shape[2]*3**2*hidden_dim}")
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            _x = x
            x = self.conv(
                features=self.ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            # print(f"{x.shape[1]*x.shape[2]*_x.shape[-1]*x.shape[-1]}")
            x = self.norm(**norm_kwargs)(x)
        if identity:
            return residual + x
        else:
            return x


class ClsUnet(nn.Module):
    num_input: int
    p_dim: int = 10
    z_dim: Tuple[int] = (8, 8, 128)
    ch: int = 128

    conv:         nn.Module = partial(nn.Conv, use_bias=False,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                      scale_init=jax.nn.initializers.ones,
                                      bias_init=jax.nn.initializers.zeros)
    lnorm:        nn.Module = partial(nn.LayerNorm, epsilon=1e-5, use_bias=True, use_scale=True,
                                      scale_init=jax.nn.initializers.ones,
                                      bias_init=jax.nn.initializers.zeros)
    relu:         Callable = nn.relu  # activation for logits and features
    silu:         Callable = nn.silu  # activation for time embedding
    relu6:        Callable = nn.activation.relu6
    fc:           nn.Module = partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)

    dsconv:         nn.Module = partial(DepthwiseSeparableConv, use_bias=False,
                                        kernel_init=jax.nn.initializers.he_normal(),
                                        bias_init=jax.nn.initializers.zeros)
    joint: int = 1  # 1: z+l -> z+l, 2: (l,z) -> l
    depth: int = 1
    version: str = "v1.0"
    droprate: float = 0
    dropout: nn.Module = partial(nn.Dropout, deterministic=False)
    input_scaling: float = 1.
    width_multi: float = 1.

    conv1x1: nn.Module = Conv1x1
    conv3x3: nn.Module = Conv3x3

    def convblock(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if self.norm.func is nn.BatchNorm else dict()
        p = p[:, None, None, :]
        t = t[:, None, None, :]
        residual = x
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x+p+t)
        x = self.norm(
            **norm_kwargs
        )(x)
        x = self.relu(x)
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x+p+t)
        x = self.norm(
            **norm_kwargs
        )(x)
        x = self.relu(x+residual)
        return x

    def input_layer(self, p, x, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if self.norm.func is nn.BatchNorm else dict()
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(
            **norm_kwargs
        )(x)
        x = self.relu(x)

        p = self.fc(
            features=self.ch,
        )(p)
        p = self.lnorm(
            **norm_kwargs
        )(p)
        p = self.relu(p)
        return p, x

    def output_layer(self, x, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if self.norm.func is nn.BatchNorm else dict()
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(
            features=self.p_dim*self.num_input
        )(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            x = self.conv(
                features=self.z_dim[-1]*self.num_input,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x)
            return p, x
        else:
            raise NotImplementedError

    @nn.compact
    def __call__(self, p, x, t, **kwargs):
        p = p/self.input_scaling
        if self.version == "v1.0":
            return self._v1_0(p, x, t, **kwargs)
        elif self.version == "v1.1":
            return self._v1_1_0(p, x, t, **kwargs)
        elif self.version == "v1.1.1":
            return self._v1_1_1(p, x, t, **kwargs)
        elif self.version == "v1.1.2":
            return self._v1_1_2(p, x, t, **kwargs)
        elif self.version == "v1.1.3":
            return self._v1_1_3(p, x, t, **kwargs)
        elif self.version == "v1.1.4":
            return self._v1_1_4(p, x, t, **kwargs)
        elif self.version == "v1.1.5":
            return self._v1_1_5(p, x, t, **kwargs)
        elif self.version == "v1.1.6":
            return self._v1_1_6(p, x, t, **kwargs)
        elif self.version == "v1.1.7":
            return self._v1_1_7(p, x, t, **kwargs)
        elif self.version == "v1.1.8":
            return self._v1_1_8(p, x, t, **kwargs)
        else:
            raise NotImplementedError

    def _v1_0(self, p, x, t, **kwargs):
        # p: logits 10*num_input (cifar10)
        # x: features (8,8,128*num_input) (resnet32x2)
        # t: time
        t = timestep_embedding(t, self.ch)
        t = self.fc(self.ch)(t)
        t = self.silu(t)
        t = self.fc(self.ch)(t)
        p, x = self.input_layer(p, x, **kwargs)
        for i in range(self.depth):
            x = self.convblock(p, x, t, **kwargs)
        return self.output_layer(x, **kwargs)

    def _v1_1_0(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        residual = x
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x+residual)
        residual = x
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x+residual)
        residual = x
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x+residual)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            return p, x

    def _v1_1_1(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        t_dim = self.ch//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=t_dim)(t)
        t = self.silu(t)
        _t = self.fc(features=self.ch)(t)
        _t = _t[:, None, None, :]
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        for _ in range(self.depth+1):
            residual = x
            _t = self.fc(features=self.ch)(t)
            _t = _t[:, None, None, :]
            x = self.conv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            x = self.conv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x+residual)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            return p, x

    def _v1_1_3(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        t_dim = self.ch//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=t_dim)(t)
        t = self.silu(t)

        p = self.fc(features=self.ch//4)(p)
        p = self.silu(p)
        p = self.fc(features=self.ch)(p)
        p = p[:, None, None, :]

        _t = self.fc(features=self.ch)(t)
        _t = _t[:, None, None, :]
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        for _ in range(self.depth+1):
            residual = x
            _t = self.fc(features=self.ch)(t)
            _t = _t[:, None, None, :]
            x = self.conv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+p+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            x = self.conv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x+residual)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            return p, x

    def _v1_1_4(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        t_dim = self.ch//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=t_dim)(t)
        t = self.silu(t)

        p = self.fc(features=self.ch//4)(p)
        p = self.silu(p)
        p = self.fc(features=self.ch//2)(p)

        _t = self.fc(features=self.ch//2)(t)
        _t = jnp.concatenate([p, _t], axis=-1)
        _t = _t[:, None, None, :]
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        if self.droprate > 0:
            x = self.dropout(self.droprate)(x)
        for _ in range(self.depth+1):
            residual = x
            _t = self.fc(features=self.ch)(t)
            _t = _t[:, None, None, :]
            x = self.dsconv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
            x = self.dsconv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x+residual)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            x = self.dsconv(
                features=self.z_dim[-1]*self.num_input,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            return p, x

    def _v1_1_5(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        t_dim = self.ch//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=t_dim)(t)
        t = self.silu(t)

        p = self.fc(features=x.shape[-1])(p)
        p = jnp.tile(p[:, None, None, :], [1, x.shape[1], x.shape[2], 1])
        x = jnp.concatenate([x, p], axis=-1)

        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        for i in range(self.depth+1):
            residual = x
            _t = self.fc(features=self.ch)(t)
            _t = _t[:, None, None, :]
            x = self.conv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            x = self.conv(
                features=(
                    self.ch*self.num_input if i == self.depth else self.ch),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            if i == self.depth:
                residual = jnp.tile(residual, [1, 1, 1, self.num_input])
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x+residual)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        if self.joint == 2:
            return p
        elif self.joint == 1:
            return p, x

    def _v1_1_6(self, p, x, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if getattr(self.norm, "func", self.norm) is nn.BatchNorm else dict()
        t_dim = self.ch//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=t_dim)(t)
        t = self.silu(t)

        p = self.fc(features=self.ch//4)(p)
        p = self.silu(p)
        p = self.fc(features=self.ch//2)(p)

        _t = self.fc(features=self.ch//2)(t)
        _t = jnp.concatenate([p, _t], axis=-1)
        _t = _t[:, None, None, :]
        x = self.conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.norm(**norm_kwargs)(x)
        x = self.relu(x)
        if self.droprate > 0:
            x = self.dropout(self.droprate)(x)
        half = (self.depth+1)//2
        for i in range(half):
            residual = x
            out_ch = self.ch if i == half-1 else self.ch//2
            _t = self.fc(features=self.ch//2)(t)
            _t = _t[:, None, None, :]
            x = self.dsconv(
                features=self.ch//2,
                kernel_size=(3, 3),
                strides=(2, 2) if i == half-1 else (1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
            x = self.conv(
                features=out_ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            if x.shape != residual.shape:
                residual = self.dsconv(
                    features=out_ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME"
                )(residual)
                residual = self.norm(**norm_kwargs)(residual)
            x = self.relu(x+residual)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
        for i in range(self.depth+1-half):
            residual = x
            _t = self.fc(features=self.ch)(t)
            _t = _t[:, None, None, :]
            x = self.dsconv(
                features=self.ch,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
            x = self.conv(
                features=self.ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME"
            )(x+_t)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu(x+residual)
            if self.droprate > 0:
                x = self.dropout(self.droprate)(x)
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        return p

    def _v1_1_7(self, p, x, t, **kwargs):
        cfgs = [
            # t, c, n, s
            (1, 32, 1, 1),
            (3, 48, 2, 2),
            (3, 64, 3, 2),
            (3, 96, 2, 1),
            (3, 128, 2, 1),
        ]
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        in_c = _divisible(x.shape[-1], 8)

        t_dim = in_c//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=in_c//2)(t)
        t = self.silu(t)

        p = self.dropout(0.2, deterministic=not kwargs["training"])(p)
        p = self.fc(features=in_c//2)(p)
        p = self.silu(p)
        p = self.dropout(0.5, deterministic=not kwargs["training"])(p)

        _t = jnp.concatenate([p, t], axis=-1)
        _x = x
        x = self.conv(
            features=in_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        # print(f"{x.shape[1]*x.shape[2]*3**2*_x.shape[-1]*x.shape[-1]}")
        x = self.norm(**norm_kwargs)(x)
        x = self.relu6(x)
        for e, c, n, s in cfgs:
            out_c = _divisible(c*self.width_multi, 2)
            _t = self.fc(features=in_c)(t)
            _t = _t[:, None, None, :]
            for i in range(n):
                if i == 0:
                    x += _t
                x = InvertedResidual(
                    in_ch=in_c,
                    ch=out_c,
                    st=s if i == 0 else 1,
                    expand=e
                )(x, **kwargs)
                in_c = out_c
        out_c = _divisible(c*self.width_multi, 2)
        _x = x
        x = Conv1x1(ch=out_c)(x, **kwargs)
        # print(f"{x.shape[1]*x.shape[2]*_x.shape[-1]*x.shape[-1]}")
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        return p

    def _v1_1_8(self, p, x, t, **kwargs):
        cfgs = [
            # t, c, n, s
            (1, 32, 1, 1),
            (3, 48, 2, 2),
            (3, 64, 3, 2),
            (3, 96, 2, 1),
            (3, 128, 2, 1),
        ]
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        in_c = _divisible(x.shape[-1], 8)

        t_dim = in_c//4  # 32
        t = timestep_embedding(t, t_dim)
        t = self.fc(features=in_c//2)(t)
        t = self.silu(t)

        # p_m = jnp.median(p, axis=-1, keepdims=True)
        # p = self.relu6(p - p_m + 6) + p_m - 6
        # p = self.refer(features=in_c//2)(p)
        # p = nn.gelu(p)
        p = FilterNet(features=in_c//2)(p)
        # p = jnp.tanh(p)
        # p = p  - 6
        # p = p[:, None, None, :]
        # p = jnp.tile(p, reps=[1,x.shape[1], x.shape[2], 1])

        t = jnp.concatenate([p, t], axis=-1)
        _x = x
        x = self.conv(
            features=in_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        # print(f"{x.shape[1]*x.shape[2]*3**2*_x.shape[-1]*x.shape[-1]}")
        x = self.norm(**norm_kwargs)(x)
        x = self.relu6(x)
        # x = jnp.concatenate([x, p], axis=-1)
        # in_c = x.shape[-1]
        for e, c, n, s in cfgs:
            out_c = _divisible(c*self.width_multi, 2)
            _t = self.fc(features=in_c)(t)
            _t = _t[:, None, None, :]
            for i in range(n):
                if i == 0:
                    x += _t
                x = InvertedResidual(
                    in_ch=in_c,
                    ch=out_c,
                    st=s if i == 0 else 1,
                    expand=e
                )(x, **kwargs)
                in_c = out_c
        out_c = _divisible(c*self.width_multi, 2)
        _x = x
        x = Conv1x1(ch=out_c)(x, **kwargs)
        # print(f"{x.shape[1]*x.shape[2]*_x.shape[-1]*x.shape[-1]}")
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(features=self.p_dim*self.num_input)(p)
        return p


def _divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)
    if new_v < 0.9*v:
        new_v += divisor
    return new_v


class FilterNet(nn.Module):
    features: int
    relu6: Callable = nn.relu6
    gelu: Callable = nn.gelu
    fc: nn.Module = partial(
        nn.Dense,
        use_bias=True,
        kernel_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.zeros
    )

    @nn.compact
    def __call__(self, x, **kwargs):
        x_m = jnp.median(x, axis=-1, keepdims=True)
        x = self.relu6(x - x_m + 6) + x_m - 6
        x = self.fc(features=self.features)(x)
        x = self.gelu(x)
        return x


class Decoder(nn.Module):
    # parameters
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: Sequence
    num_res_blocks: int
    attention_resolutions: Sequence
    dropout: float = 0.
    channel_mult: Sequence = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    # networks
    silu: Callable = nn.silu
    dense: nn.Module = nn.Dense
    embed: nn.Module = nn.Embed
    conv: nn.Module = nn.Conv
    zero_conv: nn.Module = nn.Conv
    res_block: nn.Module = ResBlock
    att_block: nn.Module = AttentionBlock
    downsample: nn.Module = Downsample
    upsample: nn.Module = Upsample
    norm: nn.Module = partial(nn.GroupNorm, num_groups=32)
    embed_sequential: nn.Module = EmbedSequential
    sequential: nn.Module = Sequential

    @nn.compact
    def __call__(self, x, **kwargs):
        if self.num_heads_upsample == -1:
            num_heads_upsample = self.num_heads
        else:
            num_heads_upsample = self.num_heads_upsample

        time_embed_dim = self.model_channels * 4
        time_embed = self.sequential([
            self.dense(time_embed_dim),
            self.silu,
            self.dense(time_embed_dim),
        ])

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        _features = ch
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        input_blocks = [
            self.sequential([
                self.conv(_features, _kernels, padding=_paddings)
            ])
        ]

        _feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                ch = int(mult * self.model_channels)
                _feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                input_block_chans.append(ch)
                ds *= 2
                _feature_size += ch

        middle_block = self.embed_sequential([
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            self.att_block(
                ch,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            self.res_block(
                ch,
                time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        ])
        _feature_size += ch

        output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    self.res_block(
                        ch + ich,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        self.att_block(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        self.res_block(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else self.upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                output_blocks.append(self.embed_sequential(layers))
                _feature_size += ch

        _features = self.out_channels
        _kernels = (3,)*self.dims
        _paddings = (1,)*self.dims
        out_fn = self.sequential([
            self.norm(),
            self.silu,
            self.zero_conv(_features, _kernels, padding=_paddings),
        ])

        # forward
        hs = []
        emb = time_embed(x)

        B = x.shape[0]
        h = x.reshape(B, 1, 1, -1)
        for module in input_blocks:
            h = module(h, emb, **kwargs)
            hs.append(h)
        h = middle_block(h, emb, **kwargs)
        for module in output_blocks:
            # h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb, **kwargs)
        h = out_fn(h)

        return h


class DiffusionClassifier(nn.Module):
    # diffusion parameters
    betas: Tuple
    n_T: int
    alpha_t: Any
    oneover_sqrta: Any
    sqrt_beta_t: Any
    alphabar_t: Any
    sqrtab: Any
    sqrtmab: Any
    mab_over_sqrtmab: Any
    sigma_weight_t: Any
    sigma_t: Any
    sigmabar_t: Any
    bigsigma_t: Any
    alpos_t: Any
    alpos_weight_t: Any
    sigma_t_square: Any
    # Classifier
    classifier: Callable
    # Unet parameters
    ver: str
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: Sequence
    num_res_blocks: int
    attention_resolutions: Sequence
    dropout: float = 0.
    channel_mult: Sequence = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    dtype: Any = jnp.float32
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    context: Sequence = None
    # scorenet
    scorenet: nn.Module = TinyUNetModel
    decodernet: nn.Module = TinyDecoder

    def setup(self):
        # self.decoder = Decoder(
        #     image_size=8,
        #     in_channels=128, # 8x8x128
        #     model_channels=self.model_channels,
        #     out_channels=128,
        #     num_res_blocks=1,
        #     attention_resolutions=(4,),
        #     dropout=0.,
        #     channel_mult=(1,1),
        #     num_heads=2,
        #     num_head_channels=-1,
        #     num_heads_upsample=-1,
        #     use_scale_shift_norm=False,
        #     resblock_updown=True,
        #     use_new_attention_order=False,
        # )
        self.decoder = self.decodernet(self.model_channels)
        self.score = self.scorenet(
            ver=self.ver,
            image_size=self.image_size,  # 8
            in_channels=self.in_channels,  # 128
            model_channels=self.model_channels,  # 256
            out_channels=self.out_channels,  # 128
            num_res_blocks=self.num_res_blocks,  # 1
            attention_resolutions=self.attention_resolutions,  # (0,)
            dropout=self.dropout,
            channel_mult=self.channel_mult,  # (1,)
            num_classes=self.num_classes,
            dtype=self.dtype,
            num_heads=self.num_heads,  # 1
            num_head_channels=self.num_head_channels,  # -1
            num_heads_upsample=self.num_heads_upsample,  # -1
            use_scale_shift_norm=self.use_scale_shift_norm,  # False
            resblock_updown=self.resblock_updown,  # False
            use_new_attention_order=self.use_new_attention_order,
            context=self.context
        )

    # def set_classifier(self, params):
    #     self.classifier = self.classifier.bind({"params":params})

    # def encode(self, x, params=None, training=True):
    #     if params is not None:
    #         out = self.classifier({"params":params}, x, training=training)
    #     out= self.classifier(x, training=training)
    #     return jax.lax.stop_gradient(out)
    def encode(self, params, z):
        return self.classifier({"params": params}, z, training=False)

    def decode(self, x, params=None):
        if params is not None:
            return self.decoder.apply({"params": params}, x, training=False)
        return self.decoder(x)

    # def __call__(self, rng, x0, z1, cls_params, stop=False, **kwargs):
    def __call__(self, rng, x0, z1, cls_params, z0=None, **kwargs):
        # x logits, z features
        if z0 is None:
            z0 = self.decode(x0)
        new_x0 = self.encode(cls_params, z0)
        z_t, t, mu_t, sigma_t = self.forward(rng, z0, z1)
        eps = self.score(z_t, t, **kwargs)
        return (eps, z_t, t, mu_t, sigma_t), new_x0, z0

    def forward(self, rng, x0, x1):
        t_rng, n_rng = jax.random.split(rng, 2)

        _ts = jax.random.randint(t_rng, (x0.shape[0],), 1, self.n_T)  # (B,)
        sigma_weight_t = self.sigma_weight_t[_ts]  # (B,)
        sigma_weight_t = expand_to_broadcast(sigma_weight_t, x0, axis=1)
        sigma_t = self.sigma_t[_ts]
        mu_t = (sigma_weight_t*x0+(1-sigma_weight_t)*x1)
        bigsigma_t = self.bigsigma_t[_ts]  # (B,)
        bigsigma_t = expand_to_broadcast(bigsigma_t, mu_t, axis=1)

        # q(X_t|X_0,X_1) = N(X_t;mu_t,bigsigma_t)
        noise = jax.random.normal(n_rng, mu_t.shape)  # (B, d)
        x_t = mu_t + noise*jnp.sqrt(bigsigma_t)
        t = _ts/self.n_T
        return x_t, t, mu_t, sigma_t


class DiffusionBridgeNetwork(nn.Module):
    base_net: nn.Module
    score_net: Sequence[nn.Module]
    cls_net: Sequence[nn.Module]
    crt_net: Sequence[nn.Module]
    dsb_stats: Any
    z_dsb_stats: Any
    # dsb_stats: dict
    # z_dsb_stats: dict
    fat: int
    joint: bool
    forget: int = 0
    temp: float = 1.
    start_temp: float = 1.
    print_inter: bool = False
    mimo_cond: bool = False
    multi_mixup: bool = False
    continuous: bool = False

    def setup(self):
        self.base = self.base_net()
        self.score = self.score_net()
        if self.cls_net is not None:
            if isinstance(self.cls_net, Sequence):
                self.cls = [cls_net() for cls_net in self.cls_net]
            else:
                self.cls = self.cls_net()
        if self.crt_net is not None:
            self.crt = self.crt_net()

    def encode(self, x, params_dict=None, **kwargs):
        # x: BxHxWxC
        multi_mixup = False
        if len(x.shape) == 5:
            # x: MxBxHxWxC, M: mutli-mixup == self.fat
            multi_mixup = True
            x = rearrange(x, "m b h w c -> (m b) h w c")
        if params_dict is not None:
            out = self.base.apply(params_dict, x, **kwargs)
        out = self.base(x, **kwargs)
        if multi_mixup:
            out = rearrange(out, "(m b) h w d -> m b h w d", m=self.fat)
        return out

    def correct(self, z, **kwargs):
        if self.crt_net is None:
            return z
        return self.crt(z)

    def classify(self, z, params_dict=None, **kwargs):
        def _classify(cls, _z):
            if params_dict is not None:
                return cls.apply(params_dict, _z, **kwargs)
            return cls(_z, **kwargs)
        if isinstance(self.cls, Sequence):
            z = rearrange(z, "(n b) h w z -> n b h w z", n=self.fat)
            logits = jnp.stack([_classify(_cls, _z)
                               for _cls, _z in zip(self.cls, z)])
            logits = rearrange(logits, "n b d -> (n b) d", n=self.fat)
            return logits
        else:
            multi_mixup = False
            if len(z.shape) == 5:
                multi_mixup = True
                z = rearrange(z, "m b h w d -> (m b) h w d")
            out = _classify(self.cls, z)
            if multi_mixup:
                out = rearrange(out, "(m b) d -> m b d", m=self.fat)
            return out

    def corrected_classify(self, z, params_dict=None, **kwargs):
        if self.fat:
            z = rearrange(
                z, "b h w (n z) -> (n b) h w z", n=self.fat)
            z = self.correct(z)
            logits = self.classify(z, params_dict, **kwargs)
            logits = rearrange(logits, "(n b) d -> n b d", n=self.fat)
        else:
            z = self.correct(z)
            logits = self.classify(z, params_dict, **kwargs)
        return logits

    def __call__(self, *args, **kwargs):
        if self.joint == 1:
            return self.joint_dbn(*args, **kwargs)
        elif self.joint == 2:
            return self.conditional_dbn(*args, **kwargs)
        else:
            return self.feature_dbn(*args, **kwargs)

    def get_joint_tensor(self, l, z):
        l = l.reshape(-1, 1, 1, l.shape[-1])
        l = jnp.tile(l, reps=[1, z.shape[1], z.shape[2], 1])
        lz = jnp.concatenate([l, z], axis=-1)
        return lz

    def get_dejoint_tensor(self, lz, sections):
        l, z = jnp.split(lz, sections, axis=-1)
        l = l[:, 0, 0, :]
        return l, z

    def set_logit(self, rng, l1):
        if self.forget == 0:
            l1 = l1/self.start_temp
        elif self.forget == 1:
            l1 = jnp.zeros_like(l1)
        elif self.forget == 2:
            l1 = jax.random.normal(rng, l1.shape)
        elif self.forget == 3:
            l1 = l1/self.start_temp
            l1 += jax.random.normal(rng, l1.shape)
        elif self.forget == 4:
            l1 = jnp.zeros_like(l1)
            label = jnp.arange(self.fat)
            onehot = common_utils.onehot(
                label, num_classes=l1.shape[-1]//self.fat)
            onehot = onehot.reshape(1, -1)
            l1 += onehot
        elif self.forget == 5:
            l1 = l1/self.start_temp
            mask = jax.random.bernoulli(rng, p=0.7, shape=l1.shape)
            l1 = mask*l1
        return l1

    def conditional_dbn(self, rng, l0, x1, base_params=None, cls_params=None, **kwargs):
        z1 = self.encode(x1, base_params, **kwargs)
        l1 = self.classify(z1, cls_params, **kwargs)/self.temp
        if self.fat and len(l1.shape) == 2:
            reps = [1]*len(l1.shape[:-1]) + [self.fat]
            l1 = jnp.tile(l1, reps)
            if self.mimo_cond or self.multi_mixup:
                reps = [1]*len(z1.shape[:-1]) + [self.fat]
                z1 = jnp.tile(z1, reps)
        elif self.fat and len(l1.shape) == 3:
            z1 = rearrange(z1, "m b h w d -> b h w (m d)")
            l1 = rearrange(l1, "m b d -> b (m d)")
        l1 = self.set_logit(rng, l1)
        l_t, t, mu_t, sigma_t, _ = self.forward(rng, l0, l1)
        if self.mimo_cond:
            _l_t = rearrange(l_t, "b (n d) -> n b d", n=self.fat)
            _z1 = rearrange(z1, "b h w (n d) -> n b h w d", n=self.fat)
            _t = jnp.tile(t[None, :], reps=[self.fat, 1])
            rngs = jax.random.split(rng, self.fat)
            perm = jax.vmap(
                partial(jax.random.permutation, x=x1.shape[0]))(rngs)
            inv_perm = jax.vmap(jnp.argsort)(perm)
            _l_t = jax.vmap(lambda x, perm: x[perm])(_l_t, perm)
            _z1 = jax.vmap(lambda x, perm: x[perm])(_z1, perm)
            _t = jax.vmap(lambda x, perm: x[perm])(_t, perm)
            _l_t = rearrange(_l_t, "n b d -> b (n d)", n=self.fat)
            _z1 = rearrange(_z1, "n b h w d -> b h w (n d)", n=self.fat)
            _t = rearrange(_t, "n b -> b n", n=self.fat)
            _eps = self.score(_l_t, _z1, _t, **kwargs)
            _eps = rearrange(_eps, "b (n d) -> n b d", n=self.fat)
            _eps = jax.vmap(lambda x, inv: x[inv])(_eps, inv_perm)
            eps = rearrange(_eps, "n b d -> b (n d)", n=self.fat)
        else:
            eps = self.score(l_t, z1, t, **kwargs)
        _sigma_t = expand_to_broadcast(sigma_t, l_t, axis=1)
        l0eps = l_t - _sigma_t*eps
        if self.fat:
            l0eps = rearrange(
                l0eps, "b (n d) -> (n b) d", n=self.fat)
            l0eps = self.correct(l0eps, **kwargs)
            l0eps = rearrange(
                l0eps, "(n b) d -> n b d", n=self.fat)
        else:
            l0eps = self.correct(l0eps, **kwargs)
        return (eps, l_t, t, mu_t, sigma_t), l0eps

    def joint_dbn(self, rng, lz0, x1, base_params=None, cls_params=None, **kwargs):
        # l: logits
        # z: features
        l0, z0 = lz0
        z1 = self.encode(x1, base_params, **kwargs)
        l1 = self.classify(z1, cls_params, **kwargs)/self.temp
        if self.fat:
            reps = [1]*len(z1.shape[:-1]) + [self.fat]
            z1 = jnp.tile(z1, reps)
            reps = [1]*len(l1.shape[:-1]) + [self.fat]
            l1 = jnp.tile(l1, reps)
        l1 = self.set_logit(rng, l1)
        lrng, zrng = jax.random.split(rng)
        l_t, t, l_mu_t, l_sigma_t, _ts = self.forward(lrng, l0, l1)
        z_t, _t, z_mu_t, z_sigma_t, _ = self.forward(
            zrng, z0, z1, _ts=_ts, dsb_stats=self.z_dsb_stats)
        leps, zeps = self.score(l_t, z_t, t, **kwargs)
        _l_sigma_t = expand_to_broadcast(l_sigma_t, l_t, axis=1)
        l0eps = l_t - _l_sigma_t*leps  # (B, cls*fat)
        if self.fat:
            l0eps = rearrange(
                l0eps, "b (n d) -> (n b) d", n=self.fat)
            l0eps = self.correct(l0eps, **kwargs)
            l0eps = rearrange(
                l0eps, "(n b) d -> n b d", n=self.fat)
        else:
            l0eps = self.correct(l0eps, **kwargs)
        return ((leps, zeps), (l_t, z_t), t, (l_mu_t, z_mu_t), (l_sigma_t, z_sigma_t)), l0eps

    def feature_dbn(self, rng, z0, x1, base_params=None, cls_params=None, **kwargs):
        z1 = self.encode(x1, base_params, **kwargs)
        if self.fat:
            reps = [1]*len(z1.shape[:-1]) + [self.fat]
            z1 = jnp.tile(z1, reps)
        z_t, t, mu_t, sigma_t = self.forward(rng, z0, z1)
        eps = self.score(z_t, t, **kwargs)
        _sigma_t = expand_to_broadcast(sigma_t, z_t, axis=1)
        z0eps = z_t - _sigma_t*eps
        logits0eps = self.corrected_classify(z0eps, cls_params, **kwargs)
        return (eps, z_t, t, mu_t, sigma_t), logits0eps

    def forward(self, rng, x0, x1, _ts=None, dsb_stats=None):
        if dsb_stats is None:
            dsb_stats = self.dsb_stats

        if self.continuous:
            coeff = dsb_stats(_ts, mode='train')
            t_rng, n_rng = jax.random.split(rng)
            if _ts is None:
                _ts = jax.random.uniform(t_rng, (x0.shape[0],), minval=0.001, maxval=1.0)
            noise = jax.random.normal(n_rng, mu_t.shape) # (B, d)
            mu_t = batch_mul(coeff['x0'], x0) + batch_mul(coeff['x1'], x1)
            x_t = mu_t + batch_mul(coeff['n'], noise)
            return x_t, _ts, mu_t, coeff['sigma_t'], _ts

        else:
            n_T = dsb_stats["n_T"]
            _sigma_weight_t = dsb_stats["sigma_weight_t"]
            _sigma_t = dsb_stats["sigma_t"]
            _bigsigma_t = dsb_stats["bigsigma_t"]

            t_rng, n_rng = jax.random.split(rng, 2)

            if _ts is None:
                _ts = jax.random.randint(t_rng, (x0.shape[0],), 1, n_T)  # (B,)
            sigma_weight_t = _sigma_weight_t[_ts]  # (B,)
            sigma_weight_t = expand_to_broadcast(sigma_weight_t, x0, axis=1)
            sigma_t = _sigma_t[_ts]
            mu_t = (sigma_weight_t*x0+(1-sigma_weight_t)*x1)
            bigsigma_t = _bigsigma_t[_ts]  # (B,)
            bigsigma_t = expand_to_broadcast(bigsigma_t, mu_t, axis=1)

            # q(X_t|X_0,X_1) = N(X_t;mu_t,bigsigma_t)
            noise = jax.random.normal(n_rng, mu_t.shape)  # (B, d)
            x_t = mu_t + noise*jnp.sqrt(bigsigma_t)
            t = _ts/n_T
            return x_t, t, mu_t, sigma_t, _ts

    def sample(self, *args, **kwargs):
        if self.joint == 1:
            return self.joint_sample(*args, **kwargs)
        elif self.joint == 2:
            return self.conditional_sample(*args, **kwargs)
        else:
            return self.feature_sample(*args, **kwargs)

    def feature_sample(self, rng, sampler, x):
        zB = self.encode(x, training=False)
        if self.fat:
            reps = [1]*len(zB.shape[:-1]) + [self.fat]
            zB = jnp.tile(zB, reps)
        zC = sampler(
            partial(self.score, training=False), rng, zB)
        if self.fat:
            zC = rearrange(
                zC, "b h w (n z) -> (n b) h w z", n=self.fat)
            zC = self.correct(zC)
            _zC = rearrange(
                zC, "(n b) h w z -> n b h w z", n=self.fat)[0]
            logitsC = self.classify(zC, training=False)
            logitsC = rearrange(
                logitsC, "(n b) d -> n b d", n=self.fat)
        else:
            zC = self.correct(zC)
            _zC = zC
            logitsC = self.classify(zC, training=False)
        return logitsC, _zC

    def joint_sample(self, rng, sampler, x):
        zB = self.encode(x, training=False)
        lB = self.classify(zB, training=False)/self.temp
        if self.fat:
            reps = [1]*len(zB.shape[:-1]) + [self.fat]
            zB = jnp.tile(zB, reps)
            reps = [1]*len(lB.shape[:-1]) + [self.fat]
            lB = jnp.tile(lB, reps)
        lB = self.set_logit(rng, lB)
        out = sampler(
            partial(self.score, training=False), rng, lB, zB)
        if self.print_inter:
            (lC, _zC), lC_arr = out
        else:
            lC, _zC = out
        if self.fat:
            lC = rearrange(lC, "b (n z) -> (n b) z", n=self.fat)
            lC = self.correct(lC)
            lC = rearrange(lC, "(n b) z -> n b z", n=self.fat)
        else:
            lC = self.correct(lC)
        if self.print_inter:
            return self.temp*lC, lC_arr
        return self.temp*lC, _zC

    def conditional_sample(self, rng, sampler, x):
        zB = self.encode(x, training=False)
        lB = self.classify(zB, training=False)
        _lB = lB/self.temp
        if self.fat and len(lB.shape) == 2:
            reps = [1]*len(_lB.shape[:-1]) + [self.fat]
            _lB = jnp.tile(_lB, reps)
            if self.mimo_cond or self.multi_mixup:
                reps = [1]*len(zB.shape[:-1]) + [self.fat]
                zB = jnp.tile(zB, reps)
        elif self.fat and len(lB.shape) == 3:
            zB = rearrange(zB, "m b h w d -> b h w (m d)")
            _lB = rearrange(_lB, "m b d -> b (m d)")
        _lB = self.set_logit(rng, _lB)
        lC = sampler(
            partial(self.score, training=False), rng, _lB, zB)
        if self.print_inter:
            lC, lC_arr = lC
        if self.fat:
            lC = rearrange(lC, "b (n z) -> (n b) z", n=self.fat)
            lC = self.correct(lC)
            lC = rearrange(lC, "(n b) z -> n b z", n=self.fat)
        else:
            lC = self.correct(lC)
        if self.print_inter:
            return self.temp*lC, lC_arr
        return self.temp*lC, lB

    def feature_dbn_deprecated(self, rng, z0, x1, base_params=None, cls_params=None, **kwargs):
        z1 = self.encode(x1, base_params, **kwargs)
        if self.fat:
            reps = [1]*len(z1.shape[:-1]) + [self.fat]
            z1 = jnp.tile(z1, reps)
        z_t, t, mu_t, sigma_t, _ = self.forward(rng, z0, z1)
        if self.fat:
            perm = jax.random.permutation(rng, z1.shape[0]*self.fat)
            _z_t = rearrange(z_t, "b h w (n z) -> (n b) h w z", n=self.fat)
            _z_t = _z_t[perm]
            inv_perm = jnp.argsort(perm)
            _z_t = rearrange(_z_t, "(n b) h w z -> b h w (n z)", n=self.fat)
        else:
            _z_t = z_t
        eps = self.score(_z_t, t, **kwargs)
        if self.fat:
            eps = rearrange(eps, "b h w (n z) -> (n b) h w z", n=self.fat)
            eps = eps[inv_perm]
            eps = rearrange(eps, "(n b) h w z -> b h w (n z)", n=self.fat)
        _sigma_t = expand_to_broadcast(sigma_t, z_t, axis=1)
        z0eps = z_t - _sigma_t*eps
        logits0eps = self.corrected_classify(z0eps, cls_params, **kwargs)
        return (eps, z_t, t, mu_t, sigma_t), logits0eps


class DoubleDSB(nn.Module):
    score_net1: nn.Module
    score_net2: nn.Module
    cls_net1: nn.Module
    cls_net2: nn.Module
    crt_net1: nn.Module
    crt_net2: nn.Module
    dsb_state: dict

    def setup(self):
        self.score1 = self.score_net1()
        self.score2 = self.score_net2()
        self.cls1 = self.cls_net1()
        self.cls2 = self.cls_net2()
        self.crt1 = self.crt_net1()
        self.crt2 = self.crt_net2()

    def correct(self, z_list, **kwargs):
        assert isinstance(z_list, list)
        z1 = self.crt1(z_list[0], **kwargs)
        z2 = self.crt2(z_list[1], **kwargs)

        return [z1, z2]

    def classify(self, z_list, **kwargs):
        assert isinstance(z_list, list)
        p1 = self.cls1(z_list[0], **kwargs)
        p2 = self.cls2(z_list[1], **kwargs)

    def __call__(self, rng, z0, z1, **kwargs):
        pass
