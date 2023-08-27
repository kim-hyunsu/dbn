from functools import partial
from typing import Any, Callable, Sequence, Dict, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import math
import numpy as np
from einops import rearrange

from utils import expand_to_broadcast, jprint
from .bridge import Decoder as TinyDecoder

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
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
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
    padding: Tuple[int]
    kernels_per_layer: int = 1

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        features = self.features
        kernel_size = self.kernel_size
        padding = self.padding
        x = nn.Conv(features=C * self.kernels_per_layer, kernel_size=kernel_size,
                    padding=padding, feature_group_count=C)(x)  # depthwise
        x = nn.Conv(features=features, kernel_size=(
            1, 1), padding='SAME')(x)  # pointwise
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
    relu:         Callable = nn.relu
    fc:           nn.Module = partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)

    def convblock(self, x, p, t, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if isinstance(self.norm, nn.Batchnorm) else dict()
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
        )(x+t)
        x = self.norm(
            **norm_kwargs
        )(x)
        x = self.relu(x+residual)
        return x

    def input_layer(self, x, p, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if isinstance(self.norm, nn.Batchnorm) else dict()
        x = self.conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x+p)
        x = self.norm(
            **norm_kwargs
        )(x)
        x = self.relu(x)

        p = self.fc(
            features=self.ch,
        )(p)
        p = self.norm(
            **norm_kwargs
        )(p)
        p = self.relu(p)
        return x, p

    def output_layer(self, x, **kwargs):
        norm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if isinstance(self.norm, nn.Batchnorm) else dict()
        p = jnp.mean(x, axis=(1, 2))
        p = self.fc(
            features=self.p_dim*self.num_input
        )(p)
        x = self.conv(
            features=self.ch*self.num_input,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        return x, p

    @nn.compact
    def __call__(self, x, p, t, **kwargs):
        # x: features (8,8,128*num_input) (resnet32x2)
        # p: logits 10*num_input (cifar10)
        # t: time
        t = timestep_embedding(t, self.ch)
        t = self.fc(self.ch)(t)
        t = self.silu(t)
        t = self.fc(self.ch)(t)
        x, p = self.input_layer(x, p, **kwargs)
        x = self.convblock(x, p, t, **kwargs)
        x, p = self.output_layer(x, **kwargs)

        return x, p


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
    score_net: nn.Module
    cls_net: nn.Module
    dsb_stats: dict
    fat: int

    def setup(self):
        self.base = self.base_net()
        self.score = self.score_net()
        if self.cls_net is not None:
            self.cls = self.cls_net()

    def encode(self, x, params=None, batch_stats=None, **kwargs):
        if params is not None:
            params_dict = dict(params=params)
            if batch_stats is not None:
                params_dict["batch_stats"] = batch_stats
            return self.base.apply(params_dict, x, **kwargs)
        return self.base(x, **kwargs)

    def classify(self, z, params=None, batch_stats=None, **kwargs):
        if params is not None:
            params_dict = dict(params=params)
            if batch_stats is not None:
                params_dict["batch_stats"] = batch_stats
            return self.cls.apply(params_dict, z, **kwargs)
        return self.cls(z, **kwargs)

    def __call__(self, rng, z0, x1, base_params=None, cls_params=None, **kwargs):
        z1 = self.encode(x1, base_params, **kwargs)
        if self.fat > 1:
            z1 = jnp.repeat(z1, self.fat, axis=-1)
        z_t, t, mu_t, sigma_t = self.forward(rng, z0, z1)
        eps = self.score(z_t, t, **kwargs)
        _sigma_t = expand_to_broadcast(sigma_t, z_t, axis=1)
        z0eps = z_t - _sigma_t*eps
        if self.fat > 1:
            z0eps = jnp.split(z0eps, self.fat, axis=-1)
            logits0eps = [self.classify(z, cls_params, **kwargs)
                          for z in z0eps]
        else:
            logits0eps = [self.classify(z0eps, cls_params, **kwargs)]
        return (eps, z_t, t, mu_t, sigma_t), logits0eps

    def forward(self, rng, x0, x1):
        n_T = self.dsb_stats["n_T"]
        _sigma_weight_t = self.dsb_stats["sigma_weight_t"]
        _sigma_t = self.dsb_stats["sigma_t"]
        _bigsigma_t = self.dsb_stats["bigsigma_t"]

        t_rng, n_rng = jax.random.split(rng, 2)

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
        return x_t, t, mu_t, sigma_t
