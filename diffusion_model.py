from easydict import EasyDict
from functools import partial
import os

from typing import Any, OrderedDict, Tuple

import flax
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax
import jaxlib

import datetime
import defaults_diffusion_model as defaults
from tabulate import tabulate
import sys
import sghmc
from giung2.data.build import build_dataloaders
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet
from models.ddpm import ContextUnet, ddpm_schedules
from collections import OrderedDict
from PIL import Image

from utils import pixelize, normalize, unnormalize


class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale

    betas: Tuple
    n_T: int
    drop_prob: float
    alpha_t: Any
    oneover_sqrta: Any
    sqrt_beta_t: Any
    alphabar_t: Any
    sqrtab: Any
    sqrtmab: Any
    mab_over_sqrtmab: Any

    def forward(self, x, training=True, **kwargs):
        c = kwargs["conditional"]  # (B,)
        rng = kwargs["rng"]
        t_rng, n_rng, c_rng = jax.random.split(rng, 3)
        _ts = jax.random.randint(t_rng, (x.shape[0],), 1, self.n_T)  # (B,)
        noise = jax.random.normal(n_rng, x.shape)  # (B, H, W, 3)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # (B, H, W, 3)
        context_mask = jax.random.bernoulli(
            c_rng, self.drop_prob, c.shape)  # (B,)
        kwargs["c"] = c
        kwargs["t"] = _ts/self.n_T  # (B,)
        kwargs["context_mask"] = context_mask
        kwargs["training"] = training
        return noise, x_t, kwargs

    def sample(self, rng, apply, n_samples, size, stats, guide_w=0.,):
        new_size = (n_samples, *size)  # (n_samples,H,W,3)
        _rng, rng = jax.random.split(rng)
        x_init = jax.random.normal(_rng, new_size)  # (n_samples,H,W,3)
        x_i = x_init
        c_i = jnp.arange(0, 10)
        multiples = n_samples//c_i.shape[0]
        c_i = jnp.tile(c_i, multiples)  # (n_samples,)

        context_mask = jnp.zeros(c_i.shape)  # (n_samples,)

        c_i = jnp.tile(c_i, 2)  # (2*n_samples,)
        context_mask = jnp.tile(context_mask, 2)  # (2*n_samples.)
        context_mask = context_mask.at[n_samples:].set(1.)  # (2*n_samples,)

        x_i_store = []

        def body_fn(i, val):
            rng, x_i = val
            idx = self.n_T - i
            _rng, rng = jax.random.split(rng)
            t_is = jnp.array([idx/self.n_T])  # (1,)
            t_is = jnp.tile(t_is, [n_samples])  # (n_samples,)

            x_i = jnp.tile(x_i, [2, 1, 1, 1])  # (2*n_samples,H,W,3)
            t_is = jnp.tile(t_is, [2])  # (2*n_samples,)

            # (n_samples, H,W,3)
            z = jnp.where(idx > 1, jax.random.normal(
                _rng, new_size), jnp.zeros(new_size))
            eps = apply(x_i, c_i, t_is, context_mask)  # (2*n_samples,H,W,3)
            eps1 = eps[:n_samples]  # (n_samples,H,W,3)
            eps2 = eps[n_samples:]  # (n_samples,H,W,3)
            eps = (1-guide_w)*eps1 - guide_w*eps2  # (n_samples,H,W,3)
            x_i = x_i[:n_samples]  # (n_samples,H,W,3)
            x_i = (
                stats["oneover_sqrta"][idx] *
                (x_i-eps*stats["mab_over_sqrtmab"][idx])
                + stats["sqrt_beta_t"][idx]*z
            )  # (n_samples,H,W,3)
            x_i = unnormalize(x_i)
            x_i = x_i*255
            x_i = jnp.clip(x_i, 0, 255)
            x_i = x_i/255
            x_i = normalize(x_i)
            return rng, x_i

        _, x_i = jax.lax.fori_loop(0, self.n_T, body_fn, (rng, x_i))

        return x_i, x_init


def get_ckpt_temp(ckpt_dir):
    if not ckpt_dir.endswith("sghmc"):
        sghmc_ckpt_dir = os.path.join(ckpt_dir, "sghmc")
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=None)
    sghmc_config = EasyDict(sghmc_ckpt["config"])
    dataloaders = build_dataloaders(sghmc_config)
    del sghmc_ckpt["model"]

    # specify precision
    model_dtype = jnp.float32
    if sghmc_config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build model
    _ResNet = partial(
        FlaxResNet,
        depth=sghmc_config.model_depth,
        widen_factor=sghmc_config.model_width,
        dtype=model_dtype,
        pixel_mean=defaults.PIXEL_MEAN,
        pixel_std=defaults.PIXEL_STD,
        num_classes=dataloaders['num_classes'])

    if sghmc_config.model_style == 'BN-ReLU':
        model = _ResNet()
    elif sghmc_config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    else:
        raise Exception("Unknown model_style")

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model_dtype))
    variables = initialize_model(jax.random.PRNGKey(sghmc_config.seed), model)

    sghmc_state = sghmc.get_sghmc_state(
        sghmc_config, dataloaders, model, variables)
    sghmc_ckpt["model"] = sghmc_state
    sghmc_ckpt["ckpt_dir"] = sghmc_ckpt_dir
    return sghmc_ckpt


def load_classifer_state(idx, template):
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=template["ckpt_dir"], target=template, step=idx
    )
    state = sghmc_ckpt["model"]
    return state, sghmc_ckpt


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    init_rng, rng = jax.random.split(rng)
    # TODO: change hyperparameters

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)
    config.n_classes = dataloaders["num_classes"]
    config.ws_test = [0.0, 0.5, 2.0]

    beta1 = 1e-4
    beta2 = 0.02
    ddpm_stats = ddpm_schedules(beta1, beta2, config.T)
    # ddpm = DDPM(
    #     nn_model=ContextUnet(
    #         in_channels=3,
    #         n_feat=config.n_feat,
    #         n_classes=config.n_classes),
    #     betas=(beta1, beta2),
    #     n_T=config.T,
    #     drop_prob=0.1,
    #     dtype=model_dtype,
    #     **ddpm_stats)
    score_func = ContextUnet(
        in_channels=3,
        n_feat=config.n_feat,
        n_classes=config.n_classes,
    )

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)
        return init({'params': key},
                    jnp.ones(dataloaders['image_shape'], model_dtype),
                    c=jnp.zeros((1,)),
                    t=jnp.zeros((1,)),
                    context_mask=jnp.zeros((1,)))
    variables = initialize_model(init_rng, score_func)

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders['trn_steps_per_epoch'])
    optimizer = optax.adam(learning_rate=scheduler)

    state = TrainState.create(
        apply_fn=score_func.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables["batch_stats"],
        rng=init_rng,
        dynamic_scale=dynamic_scale,
        betas=(beta1, beta2),
        n_T=config.T,
        drop_prob=config.drop_prob,
        **ddpm_stats)

    pixel_mean = jnp.array(defaults.PIXEL_MEAN, dtype=model_dtype)
    pixel_std = jnp.array(defaults.PIXEL_STD, dtype=model_dtype)

    def normalize(x):
        return (x-pixel_mean)/pixel_std

    def unnormalize(x):
        return pixel_mean + pixel_std*x

    def pixelize(x):
        return (x*255).astype("uint8")

    def mse_loss(noise, output):
        assert len(output.shape) == 4
        loss = jnp.sum((noise-output)**2, axis=(1, 2, 3))
        return loss

    def step_train(state, batch, config):
        x = batch["images"]
        y = batch["labels"]
        x = normalize(x)

        def loss_fn(params):
            noise, x_t, kwargs = state.forward(
                x, training=True, conditional=y, rng=state.rng)
            output, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats}, x_t, mutable="batch_stats", **kwargs)
            loss = mse_loss(noise, output)
            loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
            count = jnp.sum(batch["marker"])
            loss = jnp.sum(loss) / count

            metrics = OrderedDict({"loss": loss})

            return loss, (metrics, new_model_state)

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            dynamic_scale, is_fin, (loss, aux), grads = dynamic_scale.value_and_grad(
                loss_fn, has_aux=True, axis_name='batch')(state.params)
        else:
            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')

        metrics, new_model_state = aux
        grads = jax.tree_util.tree_map(
            lambda g, p: g + config.optim_weight_decay * p, grads, state.params)
        #### gradient clipping #######
        # norm = jax.experimental.optimizers.l2_norm(grads)
        # eps = 1e-9
        # def clip_normalize(g): return jnp.where(norm < 1, g, g*1 / (norm+eps))
        # grads = jax.tree_util.tree_map(clip_normalize, grads)
        ##############################
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state['batch_stats'])
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        metrics["lr"] = scheduler(state.step)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.params, state.params),
                dynamic_scale=dynamic_scale)

        return new_state, metrics

    def step_valid(state, batch):
        x = batch["images"]
        y = batch["labels"]
        x = normalize(x)
        noise, x_t, kwargs = state.forward(
            x, training=False, conditional=y, rng=state.rng)
        output = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats}, x_t, **kwargs)
        loss = mse_loss(noise, output)
        assert len(loss.shape) == 1
        loss = jnp.sum(jnp.where(batch["marker"], loss, jnp.zeros_like(
            loss)))/jnp.sum(batch["marker"])
        count = jnp.sum(batch["marker"])
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
        # loss = jnp.sum(loss) / count

        metrics = OrderedDict({"loss": loss, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        def apply(x_i, c_i, t_is, context_mask):
            output = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, x_i,
                                    c=c_i, t=t_is, context_mask=context_mask, training=False)
            return output
        x = batch["images"]
        y = batch["labels"]
        x = normalize(x)
        n_sample = 4*config.n_classes
        x_list = []
        for w_i, w in enumerate(config.ws_test):
            x_gen, x_init = state.sample(
                state.rng, apply, n_sample, (32, 32, 3), ddpm_stats, guide_w=w)
            x_real = jnp.empty(x_gen.shape, dtype=model_dtype)
            for k in range(config.n_classes):
                for j in range(n_sample//config.n_classes):
                    try:
                        idx = jnp.squeeze((y == k).nonzero())[j]
                    except:
                        idx = 0
                    x_real = x_real.at[k+(j*config.n_classes)].set(x[idx])

            x_all = jnp.concatenate([x_init, x_gen, x_real], axis=2)
            x_list.append(x_all)

        return x_list

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_train = jax.pmap(
        partial(step_train, config=config), axis_name="batch")
    p_step_valid = jax.pmap(step_valid, axis_name="batch")
    p_step_sample = jax.pmap(
        partial(step_sample, config=config), axis_name="batch")
    state = jax_utils.replicate(state)
    best_loss = float("inf")

    for epoch_idx, _ in enumerate(range(config.optim_ne), start=1):
        log_str = '[Epoch {:5d}/{:5d}] '.format(epoch_idx, config.optim_ne)
        data_rng, rng = jax.random.split(rng)

        train_metrics = []
        train_loader = dataloaders["dataloader"](rng=data_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        for batch_idx, batch in enumerate(train_loader, start=1):
            # TODO arange feature and input seperately
            loss_rng, rng = jax.random.split(rng)
            state.replace(rng=loss_rng)
            state, metrics = p_step_train(state, batch)
            train_metrics.append(metrics)
        train_metrics = common_utils.get_metrics(train_metrics)
        trn_summarized = {f'trn/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.mean(), train_metrics).items()}
        log_str += ', '.join(f'{k} {v:.3e}' for k, v in trn_summarized.items())

        state = state.replace(
            batch_stats=cross_replica_mean(state.batch_stats))

        valid_metrics = []
        valid_loader = dataloaders["val_loader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        for batch_idx, batch in enumerate(valid_loader, start=1):
            # TODO arange feature and input seperately
            loss_rng, rng = jax.random.split(rng)
            state.replace(rng=loss_rng)
            metrics = p_step_valid(state, batch)
            valid_metrics.append(metrics)

            if batch_idx == 1:
                oimage_array = np.array(batch["images"][0][0])
                oimage = Image.fromarray(pixelize(oimage_array), "RGB")
                oimage.save("test_original.png")

                x_list = p_step_sample(state, batch)
                for i, x_all in enumerate(x_list):
                    x_all = unnormalize(x_all)
                    p_bs, bs, H, W, P = x_all.shape
                    x_all = x_all.reshape(p_bs*bs*H, W, P)
                    image_array = np.array(x_all)
                    image = Image.fromarray(pixelize(image_array), "RGB")
                    image.save(f"sample_w{config.ws_test[i]}.png")
        valid_metrics = common_utils.get_metrics(valid_metrics)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), valid_metrics).items()}
        val_summarized['val/loss'] /= val_summarized['val/count']
        del val_summarized["val/count"]
        log_str += ', ' + \
            ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

        if config.save and best_loss > val_summarized["val/loss"]:
            test_metrics = []
            test_loader = dataloaders["test_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            for batch_idx, batch in enumerate(test_loader, start=1):
                metrics = p_step_valid(state, batch)
                loss_rng, rng = jax.random.split(rng)
                state.replace(rng=loss_rng)
                test_metrics.append(metrics)
            test_metrics = common_utils.get_metrics(test_metrics)
            tst_summarized = {
                f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), test_metrics).items()}
            test_nll = tst_summarized['tst/loss'] / tst_summarized['tst/count']
            del val_summarized["tst/count"]
            log_str += ', ' + \
                ', '.join(f'{k} {v:.3e}' for k, v in tst_summarized.items())

            best_loss = val_summarized['val/loss']

        log_str = datetime.datetime.now().strftime(
            '[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            break


model_list = [
    "./checkpoints/frn_sd2",
    "./checkpoints/frn_sd3",
    "./checkpoints/frn_sd5",
    "./checkpoints/frn_sd7",
    "./checkpoints/frn_sd11",
    "./checkpoints/frn_sd13",
    "./checkpoints/frn_sd17"
]


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_ne', default=200, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=1e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_weight_decay', default=0., type=float,
                        help='weight decay coefficient (default: 0.)')

    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--n_feat", default=128, type=int)
    parser.add_argument("--drop_prob", default=0., type=float)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    if args.save is not None:
        args.save = os.path.abspath(args.save)
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    print_fn = partial(print, flush=True)
    if args.save:
        def print_fn(s):
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
                fp.write(s + '\n')
            print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__ + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__ + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = f'Multiple local devices are detected:\n{jax.local_devices()}\n'
        log_str = datetime.datetime.now().strftime(
            '[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
