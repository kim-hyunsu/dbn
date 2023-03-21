from email.utils import parsedate_to_datetime
from functools import partial
import math
from typing import Any, Tuple, Callable, Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax
from tqdm import tqdm
import inspect


def mse_loss(logit, target):
    return jnp.mean((logit-target)**2)


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (
        (beta2 - beta1) * jnp.arange(0, T+1, dtype=jnp.float32) / T + beta1
    )
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
    }


class ResidualConvBlock(nn.Module):
    in_channels: int
    out_channels: int
    is_res: bool = False
    dtype: Any = jnp.float32
    conv1: nn.Module = partial(
        nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding=1)
    conv2: nn.Module = partial(
        nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding=1)
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
        h, w = x.shape[1]//2+1, x.shape[2]//2+1
        out = self.maxpool(x, window_shape=(h, w))
        return out


class UnetUp(nn.Module):
    in_channels: int
    out_channels: int
    convt: nn.Module = partial(
        nn.ConvTranspose, kernel_size=(2, 2), strides=(2, 2))
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


class ContextUnet(nn.Module):
    in_channels: int
    n_feat: int = 256
    n_classes: int = 10
    init_conv: nn.Module = partial(ResidualConvBlock, is_res=True)
    down1: nn.Module = UnetDown
    down2: nn.Module = UnetDown
    avgpool2d: Callable = partial(nn.avg_pool, window_shape=(8, 8))
    gelu: Callable = partial(nn.gelu, approximate=False)
    timeembed1: nn.Module = EmbedFC
    timeembed2: nn.Module = EmbedFC
    contextembed1: nn.Module = EmbedFC
    contextembed2: nn.Module = EmbedFC
    convt: nn.Module = partial(
        nn.ConvTranspose, kernel_size=(8, 8), strides=(8, 8))
    groupnorm1: nn.Module = nn.GroupNorm
    relu: Callable = nn.relu
    up1: nn.Module = UnetUp
    up2: nn.Module = UnetUp
    conv1: nn.Module = partial(
        nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding=1)
    groupnorm2: nn.Module = nn.GroupNorm
    conv2: nn.Module = partial(
        nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding=1)

    @nn.compact
    def __call__(self, x, training=True, **kwargs):
        c = kwargs["c"]  # (B,)
        t = kwargs["t"]  # (B,)
        context_mask = kwargs["context_mask"]  # (B,)
        kwargs["training"] = training

        x = self.init_conv(self.in_channels, self.n_feat)(
            x, **kwargs)  # (B,H,W,n_feat)
        down1 = self.down1(self.n_feat, self.n_feat)(
            x, **kwargs)  # (B,H//2,W//2,n_feat)
        down2 = self.down2(self.n_feat, 2*self.n_feat)(down1,
                                                       **kwargs)  # (B,H//4,W//4,2*n_feat)
        hiddenvec = self.avgpool2d(down2)  # (B,1,1,2*feat)
        hiddenvec = self.gelu(hiddenvec)

        c = nn.activation.one_hot(c, num_classes=self.n_classes)  # (B,10)
        context_mask = context_mask[:, None]  # (B,1)
        context_mask = jnp.tile(context_mask, [1, self.n_classes])  # (B,10)
        context_mask = -1*(1-context_mask)
        c = c*context_mask  # (B,10)

        cemb1 = self.contextembed1(
            self.n_classes, 2*self.n_feat
        )(c)  # (B, 256)
        cemb1 = cemb1[:, None, None, :]
        temb1 = self.timeembed1(
            1, 2*self.n_feat
        )(t)  # (B,256)
        temb1 = temb1[:, None, None, :]
        cemb2 = self.contextembed2(
            self.n_classes, self.n_feat
        )(c)  # (B,128)
        cemb2 = cemb2[:, None, None, :]
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
        )(cemb1*up1+temb1, skip=down2, **kwargs)  # (B,H//2,W//2, n_feat)
        up3 = self.up2(
            2*self.n_feat, self.n_feat
        )(cemb2*up2+temb2, skip=down1, **kwargs)  # (B,H,W,n_feat)

        out_fn = nn.Sequential([
            self.conv1(self.n_feat),
            self.groupnorm2(8),
            self.relu,
            self.conv2(self.in_channels)
        ])  # (B,H,W,in_ch)

        out = out_fn(jnp.concatenate([up3, x], axis=-1))

        return out


# class DDPM(nn.Module):
#     nn_model: nn.Module
#     betas: Tuple
#     n_T: int
#     drop_prob: float
#     alpha_t: Any
#     oneover_sqrta: Any
#     sqrt_beta_t: Any
#     alphabar_t: Any
#     sqrtab: Any
#     sqrtmab: Any
#     mab_over_sqrtmab: Any
#     dtype: Any = jnp.float32
#     loss_mse: Callable = mse_loss

#     @nn.compact
#     def __call__(self, x, training=True, **kwargs):
#         c = kwargs["conditional"]  # (B,)
#         rng = kwargs["rng"]
#         t_rng, n_rng, c_rng = jax.random.split(rng, 3)
#         _ts = jax.random.randint(t_rng, (x.shape[0],), 1, self.n_T)  # (B,)
#         noise = jax.random.normal(n_rng, x.shape)  # (B, H, W, 3)

#         x_t = (
#             self.sqrtab[_ts, None, None, None] * x
#             + self.sqrtmab[_ts, None, None, None] * noise
#         )  # (B, H, W, 3)
#         context_mask = jax.random.bernoulli(
#             c_rng, self.drop_prob, c.shape)  # (B,)
#         kwargs["c"] = c
#         kwargs["t"] = _ts/self.n_T  # (B,)
#         kwargs["context_mask"] = context_mask
#         kwargs["training"] = training
#         output = self.nn_model(x_t, **kwargs)
#         return noise, output

class CondMLP(nn.Module):
    out_ch: int
    hidden_ch: int

    init_conv: nn.Module = partial(ResidualConvBlock, is_res=True)
    down1: nn.Module = UnetDown
    down2: nn.Module = UnetDown
    avgpool2d: Callable = partial(nn.avg_pool, window_shape=(8, 8))
    embed_dense1: nn.Module = nn.Dense
    embed_dense2: nn.Module = nn.Dense

    dense1: nn.Module = nn.Dense
    dense2: nn.Module = nn.Dense
    dense3: nn.Module = nn.Dense
    dense4: nn.Module = nn.Dense
    batchnorm1: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    batchnorm2: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    batchnorm3: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    batchnorm4: nn.Module = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                    scale_init=jax.nn.initializers.ones,
                                    bias_init=jax.nn.initializers.zeros)
    gelu: Callable = partial(nn.gelu, approximate=False)

    @nn.compact
    def __call__(self, z, training=True, **kwargs):
        x = kwargs["c"]
        t = kwargs["t"]
        context_mask = kwargs["context_mask"]
        kwargs["training"] = training

        if "use_running_average" in inspect.signature(self.batchnorm1).parameters:
            if training is False:
                self.batchnorm1.keywords["use_running_average"] = True
                self.batchnorm2.keywords["use_running_average"] = True
                self.batchnorm3.keywords["use_running_average"] = True
                self.batchnorm4.keywords["use_running_average"] = True
            else:
                self.batchnorm1.keywords["use_running_average"] = False
                self.batchnorm2.keywords["use_running_average"] = False
                self.batchnorm3.keywords["use_running_average"] = False
                self.batchnorm4.keywords["use_running_average"] = False

        # conditional
        x = self.init_conv(
            3, self.hidden_ch//2)(x, **kwargs)  # (B,H,W,hidden_ch//2)
        down1 = self.down1(
            self.hidden_ch//2, self.hidden_ch//2)(x, **kwargs)  # (B,H//2,W//2,hidden_ch//2)
        down2 = self.down2(
            self.hidden_ch//2, self.hidden_ch)(down1, **kwargs)  # (B,H//4,W//4,hidden_ch)
        hiddenvec = self.avgpool2d(down2)  # (B,1,1,hidden_ch)
        hiddenvec = self.gelu(hiddenvec)

        context_mask = context_mask[:, None, None, None]  # (B,1,1,1)
        context_mask = jnp.tile(
            context_mask, [1, 1, 1, self.hidden_ch])  # (B,1,1,hidden_ch)
        context_mask = -1*(1-context_mask)
        x_embed = hiddenvec*context_mask  # (B,1,1,hidden_ch)
        x_embed = x_embed.reshape(-1, self.hidden_ch)  # (B,hidden_ch)

        t = self.embed_dense1(self.hidden_ch)(t[:, None])
        t = self.gelu(t)
        t_embed = self.embed_dense2(self.hidden_ch)(t)  # (B, hidden_ch)

        # features
        z1 = self.dense1(self.hidden_ch)(z)  # (B,hidden_ch)
        z1 = self.batchnorm1()(z1)
        z1 = self.gelu(z1)
        z2 = x_embed*z1+t_embed
        z3 = self.dense2(self.hidden_ch)(z2)
        z3 = self.batchnorm2()(z3)
        z3 = self.gelu(z3)
        z4 = self.dense3(self.hidden_ch)(z2+z3)
        z4 = self.batchnorm2()(z4)
        z4 = self.gelu(z4)
        z5 = self.dense4(self.out_ch)(z4)
        z5 = self.batchnorm2()(z5)
        z5 = self.gelu(z5)

        return z5


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

    @nn.compact
    def __call__(self, x, t, **kwargs):
        t_enc_dim = 2*self.pos_dim

        net = MLPBlock(layer_widths=list(self.decoder_layers)+[self.x_dim],
                       activate_final=False,
                       activation_fn=nn.leaky_relu)
        t_encoder = MLPBlock(layer_widths=list(self.encoder_layers)+[t_enc_dim],
                             activate_final=False,
                             activation_fn=nn.leaky_relu,
                             norm=True)
        x_encoder = MLPBlock(layer_widths=list(self.encoder_layers)+[t_enc_dim],
                             activate_final=False,
                             activation_fn=nn.leaky_relu,
                             norm=True)

        timestep_embed = EmbedTime(self.pos_dim)

        if len(x.shape) == 1:
            x = x[None, ...]

        temb = timestep_embed(t)  # (B, pos_dim)
        temb = t_encoder(temb, **kwargs)  # (B, t_enc_dim)
        xemb = x_encoder(x, **kwargs)  # (B, t_enc_dim)
        h = jnp.concatenate([xemb, temb], axis=-1)  # (B, 2*t_enc_dim)
        out = net(h)  # (B, x_dim)
        return out


def dsb_schedules(beta1, beta2, T):
    # TODO
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    t = jnp.arange(0, T+1, dtype=jnp.float32)
    tau = t/T

    beta_t = (beta2 - beta1) * tau + beta1

    sigma_t_square = 0.5*(beta2-beta1)*tau**2 + beta1*tau
    sigmabar_t_square = 0.5*(beta2-beta1)+beta1 - sigma_t_square
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
