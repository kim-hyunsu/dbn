from distutils.command.build import build
from easydict import EasyDict
from functools import partial
import os
import math

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

import defaults_dsb as defaults
from tabulate import tabulate
import sys
from giung2.data.build import build_dataloaders, _build_dataloader
from models.ddpm import ddpm_schedules, MLP
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import model_list, normalize, unnormalize, pixelize


def build_featureloaders(config):
    n_samples_each_mode = 2
    n_modes = 2
    train_logits_list = []
    train_images_list = []
    valid_logits_list = []
    valid_images_list = []
    test_logits_list = []
    test_images_list = []
    for mode_idx in range(n_modes):
        for i in tqdm(range(n_samples_each_mode)):
            with open(f"features/train_features_M{mode_idx}S{i}.npy", "rb") as f:
                train_logits = np.load(f)
            with open(f"features/train_images_M{mode_idx}S{i}.npy", "rb") as f:
                train_images = np.load(f)
            with open(f"features/valid_features_M{mode_idx}S{i}.npy", "rb") as f:
                valid_logits = np.load(f)
            with open(f"features/valid_images_M{mode_idx}S{i}.npy", "rb") as f:
                valid_images = np.load(f)
            with open(f"features/test_features_M{mode_idx}S{i}.npy", "rb") as f:
                test_logits = np.load(f)
            with open(f"features/test_images_M{mode_idx}S{i}.npy", "rb") as f:
                test_images = np.load(f)
            train_logits_list.append(train_logits)
            train_images_list.append(train_images)
            valid_logits_list.append(valid_logits)
            valid_images_list.append(valid_images)
            test_logits_list.append(test_logits)
            test_images_list.append(test_images)

    train_logits = np.concatenate(train_logits_list, axis=0)
    train_logits = jnp.array(train_logits)
    del train_logits_list
    train_images = np.concatenate(train_images_list, axis=0)
    train_images = jnp.array(train_images)
    del train_images_list
    valid_logits = np.concatenate(valid_logits_list, axis=0)
    valid_logits = jnp.array(valid_logits)
    del valid_logits_list
    valid_images = np.concatenate(valid_images_list, axis=0)
    valid_images = jnp.array(valid_images)
    del valid_images_list
    test_logits = np.concatenate(test_logits_list, axis=0)
    test_logits = jnp.array(test_logits)
    del test_logits_list
    test_images = np.concatenate(test_images_list, axis=0)
    test_images = jnp.array(test_images)
    del test_images_list

    dataloaders = dict(
        num_classes=10,
        image_shape=(1, 32, 32, 3),
        trn_steps_per_epoch=math.ceil(len(train_logits)/config.optim_bs),
        val_steps_per_epoch=math.ceil(len(valid_logits)/config.optim_bs),
        tst_steps_per_epoch=math.ceil(len(test_logits)/config.optim_bs)
    )

    dataloaders["trn_featureloader"] = partial(
        _build_dataloader,
        images=train_images,
        labels=train_logits,
        batch_size=config.optim_bs,
        shuffle=True,
        transform=None
    )
    dataloaders["val_featureloader"] = partial(
        _build_dataloader,
        images=valid_images,
        labels=valid_logits,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["tst_featureloader"] = partial(
        _build_dataloader,
        images=test_images,
        labels=test_logits,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    return dataloaders


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
        c = kwargs["conditional"]  # (B,H,W,3)
        rng = kwargs["rng"]
        t_rng, n_rng, c_rng = jax.random.split(rng, 3)
        _ts = jax.random.randint(t_rng, (x.shape[0],), 1, self.n_T)  # (B,)
        noise = jax.random.normal(n_rng, x.shape)  # (B, d)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # (B, d)
        context_mask = jax.random.bernoulli(
            c_rng, self.drop_prob, (c.shape[0],))  # (B,)
        kwargs["c"] = c
        kwargs["t"] = _ts/self.n_T  # (B,)
        kwargs["context_mask"] = context_mask
        kwargs["training"] = training
        return noise, x_t, kwargs

    def sample(self, rng, apply, size, x, stats, guide_w=0.,):
        n_samples = x.shape[0]  # B
        new_size = (n_samples, *size)  # (B, 10)
        _rng, rng = jax.random.split(rng)
        z_i = jax.random.normal(_rng, new_size)  # (B,10)
        x_i = x  # (n_samples, H, W, 3)

        context_mask = jnp.zeros((n_samples,))  # (B,)

        x_i = jnp.tile(x_i, [2, 1, 1, 1])  # (2*B, H, W, 3)
        context_mask = jnp.tile(context_mask, 2)  # (2*B,)
        context_mask = context_mask.at[n_samples:].set(1.)  # (2*B,)

        def body_fn(i, val):
            rng, z_i = val
            idx = self.n_T - i
            _rng, rng = jax.random.split(rng)
            t_is = jnp.array([idx/self.n_T])  # (1,)
            # t_is = jnp.tile(t_is, [n_samples, 1, 1, 1])  # (B,1,1,1)
            t_is = jnp.tile(t_is, [n_samples])  # (B,)

            z_i = jnp.tile(z_i, [2, 1])  # (2*B, 10)
            # t_is = jnp.tile(t_is, [2, 1, 1, 1])  # (2*B,1,1,1)
            t_is = jnp.tile(t_is, [2])  # (2*B,)

            h = jnp.where(idx > 1, jax.random.normal(
                _rng, new_size), jnp.zeros(new_size))  # (B,10)
            eps = apply(z_i, x_i, t_is, context_mask)  # (2*B,10)
            eps1 = eps[:n_samples]  # (B,10)
            eps2 = eps[n_samples:]  # (B,10)
            eps = (1-guide_w)*eps1 - guide_w*eps2  # (B,10)
            z_i = z_i[:n_samples]  # (B,10)
            z_i = (
                stats["oneover_sqrta"][idx] *
                (z_i-eps*stats["mab_over_sqrtmab"][idx])
                + stats["sqrt_beta_t"][idx]*h
            )  # (B,10)
            return rng, z_i

        _, z_i = jax.lax.fori_loop(0, self.n_T, body_fn, (rng, z_i))

        return z_i


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    init_rng, rng = jax.random.split(rng)

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    # dataloaders = build_dataloaders(config)
    dataloaders = build_featureloaders(config)
    config.n_classes = dataloaders["num_classes"]
    config.ws_test = [0.0, 0.5, 2.0]

    beta1 = 1e-4
    beta2 = 0.02
    ddpm_stats = ddpm_schedules(beta1, beta2, config.T)

    score_func = MLP(
        out_ch=config.n_classes,
        hidden_ch=config.n_feat
    )

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)
        return init({'params': key},
                    jnp.ones((1, config.n_classes), model_dtype),
                    c=jnp.ones(dataloaders["image_shape"], model_dtype),
                    t=jnp.ones((1,)),
                    context_mask=jnp.ones((1,)))
    variables = initialize_model(init_rng, score_func)

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders["trn_steps_per_epoch"])
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
        drop_prob=0.1,
        **ddpm_stats)

    def mse_loss(noise, output):
        assert len(output.shape) == 2
        loss = jnp.sum((noise-output)**2, axis=1)
        return loss

    def step_train(state, batch, config):
        x = batch["images"]
        logits = batch["labels"]
        x = normalize(x)

        def loss_fn(params):
            noise, x_t, kwargs = state.forward(
                logits, training=True, conditional=x, rng=state.rng)
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
        logits = batch["labels"]
        x = normalize(x)
        noise, x_t, kwargs = state.forward(
            logits, training=False, conditional=x, rng=state.rng)
        output = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats}, x_t, **kwargs)
        loss = mse_loss(noise, output)
        assert len(loss.shape) == 1
        loss = jnp.sum(jnp.where(batch["marker"], loss, jnp.zeros_like(
            loss)))/jnp.sum(batch["marker"])
        count = jnp.sum(batch["marker"])
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))

        metrics = OrderedDict({"loss": loss, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        def apply(z_i, x_i, t_is, context_mask):
            output = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, z_i,
                                    c=x_i, t=t_is, context_mask=context_mask, training=False)
            return output
        x = batch["images"]
        logits = batch["labels"]
        x = normalize(x)
        z_list = []
        for w_i, w in enumerate(config.ws_test):
            z_gen = state.sample(
                state.rng, apply, (config.n_classes,), x, ddpm_stats, guide_w=w)
            z_real = logits

            z_all = jnp.stack([z_gen, z_real], axis=-1)
            z_list.append(z_all)

        return z_list

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
        train_loader = dataloaders["trn_featureloader"](rng=data_rng)
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
        valid_loader = dataloaders["val_featureloader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        for batch_idx, batch in enumerate(valid_loader, start=1):
            # TODO arange feature and input seperately
            loss_rng, rng = jax.random.split(rng)
            state.replace(rng=loss_rng)
            metrics = p_step_valid(state, batch)
            valid_metrics.append(metrics)

            if batch_idx == 1:
                z_list = p_step_sample(state, batch)
                for i, z_all in enumerate(z_list):
                    image_array = jax.nn.sigmoid(z_all[0][0])
                    image_array = np.array(image_array)
                    image_array = pixelize(image_array)
                    image = Image.fromarray(image_array)
                    image.save(f"test_w{config.ws_test[i]}.png")
        valid_metrics = common_utils.get_metrics(valid_metrics)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), valid_metrics).items()}
        val_summarized['val/loss'] /= val_summarized['val/count']
        del val_summarized["val/count"]
        log_str += ', ' + \
            ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

        if config.save and best_loss > val_summarized["val/loss"]:
            test_metrics = []
            test_loader = dataloaders["tst_featureloader"](rng=None)
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


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_ne', default=200, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=1e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.0001, type=float,
                        help='weight decay coefficient (default: 0.0001)')

    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--n_feat", default=64, type=int)

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
