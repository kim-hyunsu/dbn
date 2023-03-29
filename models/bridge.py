from email.utils import parsedate_to_datetime
from functools import partial
import math
from typing import Any, Tuple, Callable, Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax
from tqdm import tqdm
import inspect

# revised from https://github.dev/JTT94/diffusion_schrodinger_bridge/tree/main/bridge/models/basic/basic.py


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
        if "use_running_average" in inspect.signature(self.batchnorm).parameters:
            if kwargs.get("training") is False:
                self.batchnorm.keywords["use_running_average"] = True
            else:
                self.batchnorm.keywords["use_running_average"] = False

        for ch in self.layer_widths[:-1]:
            x = nn.Dense(ch)(x)
            if self.norm:
                x = self.batchnorm()(x)
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

    # beta_t = (beta2 - beta1) * tau + beta1
    # sigma_t_square = 0.5*(beta2-beta1)*tau**2 + beta1*tau
    # sigmabar_t_square = 0.5*(beta2-beta1)+beta1 - sigma_t_square

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
        "bigsigma_t": bigsigma_t
    }


class EmbedFC(nn.Module):
    input_dim: int
    emb_dim: int
    dense1: nn.Module = nn.Dense
    dense2: nn.Module = nn.Dense
    gelu: Callable = partial(nn.gelu, approximate=False)

    @nn.compact
    def __call__(self, x, **kwargs):
        x = x.reshape(-1, self.input_dim)
        x = self.dense1(self.emb_dim)(x)
        x = self.gelu(x)
        out = self.dense2(self.emb_dim)(x)
        return out


class ResidualConvBlock(nn.Module):
    in_channels: int
    out_channels: int
    is_res: bool = False
    dtype: Any = jnp.float32
    conv1: nn.Module = partial(
        nn.Conv, kernel_size=3, strides=1, padding=1)
    conv2: nn.Module = partial(
        nn.Conv, kernel_size=3, strides=1, padding=1)
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
        if "use_running_average" in inspect.signature(self.batchnorm1).parameters:
            if training is False:
                self.batchnorm1.keywords["use_running_average"] = True
                self.batchnorm2.keywords["use_running_average"] = True
            else:
                self.batchnorm1.keywords["use_running_average"] = False
                self.batchnorm2.keywords["use_running_average"] = False

        block1 = nn.Sequential([
            self.conv1(self.out_channels),
            self.batchnorm1(),
            self.gelu
        ])
        block2 = nn.Sequential([
            self.conv2(self.out_channels),
            self.batchnorm2(),
            self.gelu
        ])

        if self.is_res:
            x1 = block1(x)  # (B,H,W,out_ch)
            x2 = block2(x1)  # (B,H,W,out_ch)
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
    conv: nn.Module = ResidualConvBlock
    # maxpool: Callable = partial(nn.max_pool, window_shape=(2, 2))
    maxpool: Callable = nn.max_pool

    @nn.compact
    def __call__(self, x, **kwargs):
        x = self.conv(self.in_channels, self.out_channels)(
            x, **kwargs)  # (B, H, W, out_ch)
        w = x.shape[1]//2+1
        out = self.maxpool(x, window_shape=w)
        return out


class UnetUp(nn.Module):
    in_channels: int
    out_channels: int
    convt: nn.Module = partial(
        nn.ConvTranspose, kernel_size=2, strides=2)
    conv1: nn.Module = ResidualConvBlock
    conv2: nn.Module = ResidualConvBlock

    @nn.compact
    def __call__(self, x, **kwargs):
        skip = kwargs["skip"]
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.convt(self.out_channels)(x)
        x = self.conv1(self.out_channels, self.out_channels)(x, **kwargs)
        out = self.conv2(self.out_channels, self.out_channels)(x, **kwargs)
        return out


class FeatureUnet(nn.Module):
    in_channels: int
    n_feat: int = 256
    init_conv: nn.Module = partial(ResidualConvBlock, is_res=True)
    down1: nn.Module = UnetDown
    down2: nn.Module = UnetDown
    avgpool1d: Callable = partial(nn.avg_pool, window_shape=8)
    gelu: Callable = partial(nn.gelu, approximate=False)
    timeembed1: nn.Module = EmbedFC
    timeembed2: nn.Module = EmbedFC
    contextembed1: nn.Module = EmbedFC
    contextembed2: nn.Module = EmbedFC
    convt: nn.Module = partial(
        nn.ConvTranspose, kernel_size=8, strides=8)
    groupnorm1: nn.Module = nn.GroupNorm
    relu: Callable = nn.relu
    up1: nn.Module = UnetUp
    up2: nn.Module = UnetUp
    conv1: nn.Module = partial(
        nn.Conv, kernel_size=3, strides=1, padding=1)
    groupnorm2: nn.Module = nn.GroupNorm
    conv2: nn.Module = partial(
        nn.Conv, kernel_size=3, strides=1, padding=1)

    @nn.compact
    def __call__(self, x, t, **kwargs):

        x = self.init_conv(self.in_channels, self.n_feat)(
            x, **kwargs)  # (B,H,W,n_feat)
        down1 = self.down1(self.n_feat, self.n_feat)(
            x, **kwargs)  # (B,H//2,W//2,n_feat)
        down2 = self.down2(self.n_feat, 2*self.n_feat)(down1,
                                                       **kwargs)  # (B,H//4,W//4,2*n_feat)
        hiddenvec = self.avgpool2d(down2)  # (B,1,1,2*feat)
        hiddenvec = self.gelu(hiddenvec)

        temb1 = self.timeembed1(
            1, 2*self.n_feat
        )(t)  # (B,256)
        temb1 = temb1[:, None, None, :]
        temb2 = self.timeembed2(
            1, self.n_feat
        )(t)  # (B,128)
        temb2 = temb2[:, None, None, :]

        up0 = nn.Sequential([
            self.convt(2*self.n_feat),
            self.groupnorm1(8),
            self.relu
        ])

        up1 = up0(hiddenvec)  # (B,H//4,W//4,2*n_feat)
        up2 = self.up1(
            4*self.n_feat, self.n_feat
        )(up1+temb1, skip=down2, **kwargs)  # (B,H//2,W//2, n_feat)
        up3 = self.up2(
            2*self.n_feat, self.n_feat
        )(up2+temb2, skip=down1, **kwargs)  # (B,H,W,n_feat)

        out_fn = nn.Sequential([
            self.conv1(self.n_feat),
            self.groupnorm2(8),
            self.relu,
            self.conv2(self.in_channels)
        ])  # (B,H,W,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out
