from functools import partial
import math
from typing import Any, Tuple, Callable, Sequence, Union
import flax.linen as nn
import jax.numpy as jnp
import jax

from utils import expand_to_broadcast

# revised from https://github.dev/JTT94/diffusion_schrodinger_bridge/tree/main/bridge/models/basic/basic.py


class CorrectionModel(nn.Module):
    layers: int

    dense: nn.Module = partial(
        nn.Dense,
        bias_init=jax.nn.initializers.zeros)
    dense2: nn.Module = nn.Dense
    swish: Callable = nn.swish

    @nn.compact
    def __call__(self, x, **kwargs):
        x_dim = x.shape[-1]
        if self.layers <= 1:
            def kernel_init(key, shape, dtype): return jnp.eye(
                x_dim, dtype=dtype)
            x = self.dense(x_dim, kernel_init=kernel_init)(x)
        else:
            for _ in range(self.layers-1):
                x = self.swish(self.dense(x_dim)(x))
            x = self.dense(x_dim)(x)
        return x


class MLPBlock(nn.Module):
    layer_widths: list
    activate_final: bool = False
    activation_fn: Callable = nn.relu
    norm: bool = False
    batchnorm: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                   scale_init=jax.nn.initializers.ones,
                                   bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):
        training = kwargs["training"]
        for ch in self.layer_widths[:-1]:
            x = nn.Dense(ch)(x)
            if self.norm:
                x = self.batchnorm(use_running_average=not training)(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.layer_widths[-1])(x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class EmbedTime(nn.Module):
    embedding_dim: int = 128

    @nn.compact
    def __call__(self, time_steps):
        assert len(time_steps.shape) == 1
        half_dim = self.embedding_dim//2
        emb = math.log(10000) / (half_dim-1)
        # (half_dim,)
        emb = jnp.exp(-emb*jnp.arange(half_dim, dtype=jnp.float32))

        emb = time_steps[..., None] * emb[None, ...]  # (B, half_dim)
        emb = jnp.concatenate(
            [jnp.sin(emb), jnp.cos(emb)], axis=-1)  # (B, embedding_dim)
        if self.embedding_dim % 2 == 1:
            assert len(emb.shape) == 2
            emb = jnp.pad(emb, [[0, 0], [0, 1]])

        return emb


class MLP(nn.Module):
    x_dim: int
    pos_dim: int
    encoder_layers: list
    decoder_layers: list
    gelu: Callable = partial(nn.gelu, approximate=False)
    tanh: Callable = nn.tanh

    @nn.compact
    def __call__(self, x, t, **kwargs):
        t_enc_dim = 2*self.pos_dim

        t_encoder = MLPBlock(layer_widths=list(self.encoder_layers)+[t_enc_dim],
                             activate_final=False,
                             activation_fn=self.gelu,
                             norm=False)
        x_encoder = MLPBlock(layer_widths=list(self.encoder_layers)+[t_enc_dim],
                             activate_final=False,
                             activation_fn=self.gelu,
                             norm=False)
        net = MLPBlock(layer_widths=list(self.decoder_layers)+[self.x_dim],
                       activate_final=False,
                       activation_fn=self.gelu,
                       norm=False)

        timestep_embed = EmbedTime(self.pos_dim)

        if len(x.shape) == 1:
            x = x[None, ...]

        temb = timestep_embed(t)  # (B, pos_dim)
        temb = t_encoder(temb, **kwargs)  # (B, t_enc_dim)
        xemb = x_encoder(x, **kwargs)  # (B, t_enc_dim)
        h = jnp.concatenate([xemb, temb], axis=-1)  # (B, 2*t_enc_dim)
        out = net(h, **kwargs)  # (B, x_dim)
        return out


def dsb_schedules(beta1, beta2, T):
    # TODO
    # assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    t = jnp.arange(0, T+1, dtype=jnp.float32)
    tau = t/T

    beta_t = jnp.where(
        tau < 0.5,
        2*(beta2-beta1)*tau+beta1,
        2*(beta1-beta2)*tau+2*beta2-beta1
    )
    sigma_t_square = jnp.where(
        tau < 0.5,
        (beta2-beta1)*tau**2 + beta1*tau,
        0.5*(beta1-beta2) + (beta1-beta2)*tau**2+(2*beta2-beta1)*tau
    )
    sigmabar_t_square = 0.5*(beta1+beta2) - sigma_t_square

    sigmabar_t_square = nn.relu(sigmabar_t_square)
    sigma_t = jnp.sqrt(sigma_t_square)
    sigmabar_t = jnp.sqrt(sigmabar_t_square)
    sigma_weight_t = sigmabar_t_square/(sigma_t_square+sigmabar_t_square)
    bigsigma_t = sigma_t_square*sigma_weight_t

    sqrt_beta_t = jnp.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    log_alphabar_t = jnp.cumsum(log_alpha_t, axis=0)
    alphabar_t = jnp.exp(log_alphabar_t)

    sqrtab = jnp.sqrt(alphabar_t)
    oneover_sqrta = 1 / jnp.sqrt(alpha_t)

    sqrtmab = jnp.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    # posterior coefficients
    alpos_t = sigma_t_square[1:] - sigma_t_square[:-1]
    alpos_t = jnp.concatenate([jnp.zeros(1), alpos_t])
    sigma_t_square_minus_zero = jnp.concatenate(
        [jnp.zeros(1), sigma_t_square[:-1]])
    alpos_weight_t = alpos_t / (alpos_t + sigma_t_square_minus_zero)

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
        "sigma_t": sigma_t,
        "sigmabar_t": sigmabar_t,
        "sigma_weight_t": sigma_weight_t,
        "bigsigma_t": bigsigma_t,
        "alpos_t": alpos_t,
        "alpos_weight_t": alpos_weight_t,
        "sigma_t_square": sigma_t_square
    }


class EmbedFC(nn.Module):
    input_dim: Union[int, Sequence]
    emb_dim: int
    dense1: nn.Module = nn.Dense
    dense2: nn.Module = nn.Dense
    gelu: Callable = partial(nn.gelu, approximate=False)
    conv: nn.Module = nn.Conv

    @nn.compact
    def __call__(self, x, **kwargs):
        input_dim = self.input_dim
        if isinstance(self.input_dim, Tuple):
            H, W, C = self.input_dim
            x = self.conv(C, (5, 5))(x)
            x = self.conv(C, (5, 5))(x)
            B, H1, W1, C1 = x.shape
            input_dim = H1*W1*C1
        x = x.reshape(-1, input_dim)
        x = self.dense1(self.emb_dim)(x)
        x = self.gelu(x)
        out = self.dense2(self.emb_dim)(x)
        return out


class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    is_res: bool = False
    dtype: Any = jnp.float32
    dense1: nn.Module = nn.Dense
    dense2: nn.Module = nn.Dense
    batchnorm1: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    batchnorm2: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    gelu: Callable = partial(nn.gelu, approximate=False)

    @nn.compact
    def __call__(self, x, **kwargs):
        training = kwargs["training"]
        same_channels = (self.in_channels == self.out_channels)
        block1 = nn.Sequential([
            self.dense1(self.out_channels),
            self.batchnorm1(use_running_average=not training),
            self.gelu
        ])
        block2 = nn.Sequential([
            self.dense2(self.out_channels),
            self.batchnorm2(use_running_average=not training),
            self.gelu
        ])

        if self.is_res:
            x1 = block1(x)
            x2 = block2(x1)
            if same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = block1(x)
            x2 = block2(x1)
            return x2


class UnetDown(nn.Module):
    in_channels: int
    out_channels: int
    block: nn.Module = ResidualBlock
    gelu: Callable = partial(nn.gelu, approximate=False)

    @nn.compact
    def __call__(self, x, **kwargs):
        out = self.block(self.in_channels, self.out_channels)(
            x, **kwargs)  # (B, out_ch)
        return out


class UnetUp(nn.Module):
    in_channels: int
    out_channels: int
    block1: nn.Module = ResidualBlock
    block2: nn.Module = ResidualBlock
    gelu: Callable = partial(nn.gelu, approximate=False)

    @nn.compact
    def __call__(self, x, **kwargs):
        skip = kwargs["skip"]
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.block1(self.in_channels, self.out_channels)(x, **kwargs)
        out = self.block2(self.out_channels, self.out_channels)(x, **kwargs)
        return out


class FeatureUnet(nn.Module):
    # arguments
    in_channels: int
    ver: str
    n_feat: int = 256
    droprate: float = 0
    context: Union[int, Sequence] = 0
    conflict: bool = False

    # networks
    init_block: nn.Module = partial(ResidualBlock, is_res=True)
    down1: nn.Module = UnetDown
    down2: nn.Module = UnetDown
    down3: nn.Module = nn.Dense
    gelu: Callable = partial(nn.gelu, approximate=False)
    timeembed0: nn.Module = EmbedTime
    timeembed1: nn.Module = EmbedFC
    timeembed2: nn.Module = EmbedFC
    ctxembed1: nn.Module = EmbedFC
    ctxembed2: nn.Module = EmbedFC
    cflembed1: nn.Module = EmbedFC
    cflembed2: nn.Module = EmbedFC
    layernorm1: nn.Module = nn.LayerNorm
    relu: Callable = nn.relu
    tanh: Callable = nn.tanh
    up0: nn.Module = nn.Dense
    up1: nn.Module = UnetUp
    up2: nn.Module = UnetUp
    dense1: nn.Module = nn.Dense
    layernorm2: nn.Module = nn.LayerNorm
    dense2: nn.Module = nn.Dense
    dropout: nn.Module = nn.Dropout

    @nn.compact
    def __call__(self, x, t, **kwargs):
        if self.ver == "v1.0":
            return self._call_v1_0(x, t, **kwargs)
        elif self.ver == "v1.1":
            return self._call_v1_1(x, t, **kwargs)
        elif self.ver == "v1.2":
            return self._call_v1_2(x, t, **kwargs)
        elif self.ver == "v1.3":
            return self._call_v1_3(x, t, **kwargs)
        elif self.ver == "v1.4":
            return self._call_v1_4(x, t, **kwargs)
        elif self.ver == "v1.5":
            return self._call_v1_5(x, t, **kwargs)
        elif self.ver == "v1.6":
            return self._call_v1_6(x, t, **kwargs)
        elif self.ver == "v1.7":
            return self._call_v1_7(x, t, **kwargs)
        elif self.ver == "v1.8":
            return self._call_v1_8(x, t, **kwargs)
        else:
            raise Exception("Invalid FeatureUnet Version")

    def _call_v1_0(self, x, t, **kwargs):
        training = kwargs["training"]

        x = self.init_block(
            x.shape[-1], self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        x = self.dropout(self.droprate/2, deterministic=not training)(x)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.dropout(self.droprate, deterministic=not training)(down1)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        down2 = self.dropout(self.droprate, deterministic=not training)(down2)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)
        hiddenvec = self.dropout(
            self.droprate, deterministic=not training)(hiddenvec)

        temb1 = self.timeembed1(
            1, self.n_feat
        )(t)  # (B,n_feat)
        temb2 = self.timeembed2(
            1, self.n_feat
        )(t)  # (B,n_feat)

        cemb1 = 1
        cemb2 = 1
        if self.context:
            c = kwargs["ctx"]
            cemb1 = self.ctxembed1(
                self.context, self.n_feat
            )(c)
            cemb2 = self.ctxembed2(
                self.context, self.n_feat
            )(c)

        up0 = nn.Sequential([
            self.up0(self.n_feat),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat)
        up1_input = cemb1*up1+temb1
        feat_input = self.n_feat*2
        up2 = self.up1(
            feat_input, self.n_feat
        )(up1_input, skip=down2, **kwargs)  # (B,n_feat)
        up2_input = cemb2*up2+temb2
        feat_input = self.n_feat*2
        up3 = self.up2(
            feat_input, self.n_feat
        )(up2_input, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_1(self, x, t, **kwargs):
        x = self.init_block(
            self.in_channels, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.down1(
            self.n_feat, self.n_feat//2
        )(x, **kwargs)  # (B,n_feat//2)
        down2 = self.down2(
            self.n_feat//2, self.n_feat//4
        )(down1, **kwargs)  # (B,n_feat//4)
        hiddenvec = self.down3(
            self.n_feat//4
        )(down2)  # (B,n_feat//4)
        hiddenvec = self.gelu(hiddenvec)

        temb01 = self.timeembed0(
            self.n_feat//4
        )(t)  # (B, n_feat//4)
        temb02 = self.timeembed0(
            self.n_feat//2
        )(t)  # (B, n_feat//2)
        temb1 = self.timeembed1(
            1, self.n_feat//4
        )(t)  # (B,n_feat//4)
        temb2 = self.timeembed2(
            1, self.n_feat//2
        )(t)  # (B,n_feat//2)

        up0 = nn.Sequential([
            self.up0(self.n_feat//4),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat//4)
        up2_input = jnp.concatenate([up1, temb1], axis=-1)
        up2_skip = jnp.concatenate([down2, temb01], axis=-1)
        up2 = self.up1(
            self.n_feat, self.n_feat//2
        )(up2_input, skip=up2_skip, **kwargs)  # (B,n_feat//2)
        up3_input = jnp.concatenate([up2, temb2], axis=-1)
        up3_skip = jnp.concatenate([down1, temb02], axis=-1)
        up3 = self.up2(
            self.n_feat*2, self.n_feat
        )(up3_input, skip=up3_skip, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_2(self, x, t, **kwargs):
        x = self.init_block(
            self.in_channels, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)

        temb0 = self.timeembed0(
            self.n_feat
        )(t)  # (B, n_feat)
        temb1 = self.timeembed1(
            self.n_feat, self.n_feat//2
        )(temb0)  # (B,n_feat//2)
        temb2 = self.timeembed2(
            self.n_feat, self.n_feat//2
        )(temb0)  # (B,n_feat//2)

        up0 = nn.Sequential([
            self.up0(self.n_feat//2),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat//2)

        up2_input = jnp.concatenate([up1, temb1], axis=-1)
        up2 = self.up1(
            self.n_feat*2, self.n_feat//2
        )(up2_input, skip=down2, **kwargs)  # (B,n_feat//2)
        up3_input = jnp.concatenate([up2, temb2], axis=-1)
        up3 = self.up2(
            self.n_feat*2, self.n_feat
        )(up3_input, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_3(self, x, t, **kwargs):
        x = self.init_block(
            self.in_channels, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)

        temb1 = self.timeembed1(
            1, self.n_feat
        )(t)  # (B,n_feat)
        temb2 = self.timeembed2(
            1, self.n_feat
        )(t)  # (B,n_feat)

        up0 = nn.Sequential([
            self.up0(self.n_feat),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat)
        up2_input = jnp.concatenate([up1, temb1], axis=-1)
        up2 = self.up1(
            self.n_feat*3, self.n_feat
        )(up2_input, skip=down2, **kwargs)  # (B,n_feat)
        up3_input = jnp.concatenate([up2, temb2], axis=-1)
        up3 = self.up2(
            self.n_feat*3, self.n_feat
        )(up3_input, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_4(self, x, t, **kwargs):
        x = self._call_v1_0(x, t, **kwargs)
        x = self.tanh(x)
        return x

    def _call_v1_5(self, x, t, **kwargs):
        training = kwargs["training"]

        x = self.init_block(
            self.in_channels, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        x = self.dropout(self.droprate/2, deterministic=not training)(x)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.dropout(self.droprate, deterministic=not training)(down1)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        down2 = self.dropout(self.droprate, deterministic=not training)(down2)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)
        hiddenvec = self.dropout(
            self.droprate, deterministic=not training)(hiddenvec)

        temb1 = self.timeembed1(
            1, self.n_feat
        )(t)  # (B,n_feat)
        temb2 = self.timeembed2(
            1, self.n_feat
        )(t)  # (B,n_feat)

        cemb1 = 1
        cemb2 = 1
        if self.context:
            c = kwargs["ctx"]
            cemb1 = self.ctxembed1(
                self.context, self.n_feat
            )(c)
            cemb2 = self.ctxembed2(
                self.context, self.n_feat
            )(c)

        up0 = nn.Sequential([
            self.up0(self.n_feat),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat)
        input_up1 = jnp.concatenate(
            [up1+temb1, cemb1], axis=-1) if self.context else up1+temb1
        up2 = self.up1(
            self.n_feat*2 if not self.context else 3*self.n_feat, self.n_feat
        )(input_up1, skip=down2, **kwargs)  # (B,n_feat)
        input_up2 = jnp.concatenate(
            [up2+temb2, cemb2], axis=-1) if self.context else up2+temb2
        up3 = self.up2(
            self.n_feat*2 if not self.context else 3*self.n_feat, self.n_feat
        )(input_up2, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_6(self, x, t, **kwargs):
        training = kwargs["training"]

        x = self.init_block(
            self.in_channels, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        x = self.dropout(self.droprate/2, deterministic=not training)(x)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.dropout(self.droprate, deterministic=not training)(down1)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        down2 = self.dropout(self.droprate, deterministic=not training)(down2)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)
        hiddenvec = self.dropout(
            self.droprate, deterministic=not training)(hiddenvec)

        if self.conflict:
            cf = kwargs["cfl"]
            temb1 = self.timeembed1(
                2, self.n_feat//2 if self.context else self.n_feat
            )(jnp.stack([t, cf], axis=-1))  # (B,n_feat)
            temb2 = self.timeembed2(
                2, self.n_feat//2 if self.context else self.n_feat
            )(jnp.stack([t, cf], axis=-1))  # (B,n_feat)
        else:
            temb1 = self.timeembed1(
                1, self.n_feat//2 if self.context else self.n_feat
            )(t)  # (B,n_feat)
            temb2 = self.timeembed2(
                1, self.n_feat//2 if self.context else self.n_feat
            )(t)  # (B,n_feat)

        cemb1 = 1
        cemb2 = 1
        if self.context:
            c = kwargs["ctx"]
            cemb1 = self.ctxembed1(
                self.context, self.n_feat//2
            )(c)
            cemb2 = self.ctxembed2(
                self.context, self.n_feat//2
            )(c)
            emb1 = jnp.concatenate([temb1, cemb1], axis=-1)
            emb2 = jnp.concatenate([temb2, cemb2], axis=-1)
        else:
            emb1 = temb1
            emb2 = temb2

        up0 = nn.Sequential([
            self.up0(self.n_feat),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat)
        up2 = self.up1(
            self.n_feat*2, self.n_feat
        )(emb1*up1, skip=down2, **kwargs)  # (B,n_feat)
        up3 = self.up2(
            self.n_feat*2, self.n_feat
        )(emb2*up2, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out

    def _call_v1_7(self, x, t, **kwargs):
        if self.conflict:
            cf = kwargs["cfl"]-0.5
            cf = expand_to_broadcast(cf, x, axis=1)
            x = jnp.concatenate([x,cf], axis=-1)
        return self._call_v1_0(x, t, **kwargs)

    def _call_v1_8(self, x, t, **kwargs):
        training = kwargs["training"]

        t = expand_to_broadcast(t, x, axis=1)
        x = jnp.concatenate([x, t], axis=-1)
        if self.conflict:
            cf = kwargs["cfl"]-0.5
            cf = expand_to_broadcast(cf, x, axis=1)
            x = jnp.concatenate([x,cf], axis=-1)
            
        x = self.init_block(
            x.shape[-1], self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        x = self.dropout(self.droprate/2, deterministic=not training)(x)
        down1 = self.down1(
            self.n_feat, self.n_feat
        )(x, **kwargs)  # (B,n_feat)
        down1 = self.dropout(self.droprate, deterministic=not training)(down1)
        down2 = self.down2(
            self.n_feat, self.n_feat
        )(down1, **kwargs)  # (B,n_feat)
        down2 = self.dropout(self.droprate, deterministic=not training)(down2)
        hiddenvec = self.down3(
            self.n_feat
        )(down2)  # (B,n_feat)
        hiddenvec = self.gelu(hiddenvec)
        hiddenvec = self.dropout(
            self.droprate, deterministic=not training)(hiddenvec)

        cemb1 = 1
        cemb2 = 1
        if self.context:
            c = kwargs["ctx"]
            cemb1 = self.ctxembed1(
                self.context, self.n_feat
            )(c)
            cemb2 = self.ctxembed2(
                self.context, self.n_feat
            )(c)

        up0 = nn.Sequential([
            self.up0(self.n_feat),
            self.layernorm1(),
            self.gelu
        ])

        up1 = up0(hiddenvec)  # (B,n_feat)
        up1_input = cemb1*up1
        feat_input = self.n_feat*2
        up2 = self.up1(
            feat_input, self.n_feat
        )(up1_input, skip=down2, **kwargs)  # (B,n_feat)
        up2_input = cemb2*up2
        feat_input = self.n_feat*2
        up3 = self.up2(
            feat_input, self.n_feat
        )(up2_input, skip=down1, **kwargs)  # (B, n_feat)

        out_fn = nn.Sequential([
            self.dense1(self.n_feat),
            self.layernorm2(),
            self.gelu,
            self.dense2(self.in_channels)
        ])  # (B,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out
