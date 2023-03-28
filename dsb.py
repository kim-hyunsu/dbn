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
from models.ddpm import dsb_schedules, MLP
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import model_list, normalize, unnormalize, pixelize, normalize_logits, unnormalize_logits


def build_featureloaders(config):
    # B: current mode, A: other modes
    n_Amodes = 1
    n_samples_each_mode = 1
    n_samples_each_Bmode = n_Amodes*n_samples_each_mode
    n_samples_each_Amode = n_samples_each_mode
    train_Alogits_list = []
    train_Blogits_list = []
    valid_Alogits_list = []
    valid_Blogits_list = []
    test_Alogits_list = []
    test_Blogits_list = []
    for mode_idx in range(1+n_Amodes):  # total 1+n_Amodes
        if mode_idx == 0:  # p_B
            for i in tqdm(range(n_samples_each_Bmode)):
                with open(f"features/train_features_M{mode_idx}S{i}.npy", "rb") as f:
                    train_logits = np.load(f)
                with open(f"features/valid_features_M{mode_idx}S{i}.npy", "rb") as f:
                    valid_logits = np.load(f)
                with open(f"features/test_features_M{mode_idx}S{i}.npy", "rb") as f:
                    test_logits = np.load(f)
                train_Blogits_list.append(train_logits)
                valid_Blogits_list.append(valid_logits)
                test_Blogits_list.append(test_logits)
        else:  # p_A (mixture of modes)
            for i in tqdm(range(n_samples_each_Amode)):
                with open(f"features/train_features_M{mode_idx}S{i}.npy", "rb") as f:
                    train_logits = np.load(f)
                with open(f"features/valid_features_M{mode_idx}S{i}.npy", "rb") as f:
                    valid_logits = np.load(f)
                with open(f"features/test_features_M{mode_idx}S{i}.npy", "rb") as f:
                    test_logits = np.load(f)
                train_Alogits_list.append(train_logits)
                valid_Alogits_list.append(valid_logits)
                test_Alogits_list.append(test_logits)

    train_logitsA = np.concatenate(train_Alogits_list, axis=0)
    train_logitsA = jnp.array(train_logitsA)
    del train_Alogits_list

    train_logitsB = np.concatenate(train_Blogits_list, axis=0)
    train_logitsB = jnp.array(train_logitsB)
    del train_Blogits_list

    valid_logitsA = np.concatenate(valid_Alogits_list, axis=0)
    valid_logitsA = jnp.array(valid_logitsA)
    del valid_Alogits_list

    valid_logitsB = np.concatenate(valid_Blogits_list, axis=0)
    valid_logitsB = jnp.array(valid_logitsB)
    del valid_Blogits_list

    test_logitsA = np.concatenate(test_Alogits_list, axis=0)
    test_logitsA = jnp.array(test_logitsA)
    del test_Alogits_list

    test_logitsB = np.concatenate(test_Blogits_list, axis=0)
    test_logitsB = jnp.array(test_logitsB)
    del test_Blogits_list

    dataloaders = dict(
        num_classes=10,
        image_shape=(1, 32, 32, 3),
        trn_steps_per_epoch=math.ceil(len(train_logitsA)/config.optim_bs),
        val_steps_per_epoch=math.ceil(len(valid_logitsA)/config.optim_bs),
        tst_steps_per_epoch=math.ceil(len(test_logitsA)/config.optim_bs)
    )
    dataloaders["featureloader"] = partial(
        _build_dataloader,
        images=train_logitsB,
        labels=train_logitsA,
        batch_size=config.optim_bs,
        shuffle=True,
        transform=None
    )

    dataloaders["trn_featureloader"] = partial(
        _build_dataloader,
        images=train_logitsB,
        labels=train_logitsA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["val_featureloader"] = partial(
        _build_dataloader,
        images=valid_logitsB,
        labels=valid_logitsA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["tst_featureloader"] = partial(
        _build_dataloader,
        images=test_logitsB,
        labels=test_logitsA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    # count = 0
    # sum = 0
    # for batch in dataloaders["trn_featureloader"](rng=None):
    #     logitsB = batch["images"]
    #     logitsA = batch["labels"]
    #     logitsB = jnp.where(
    #         batch["marker"][..., None], logitsB, jnp.zeros_like(logitsB))
    #     logitsA = jnp.where(
    #         batch["marker"][..., None], logitsA, jnp.zeros_like(logitsA))
    #     sum += jnp.sum(logitsB, [0, 1])+jnp.sum(logitsA, [0, 1])
    #     count += 2*batch["marker"].sum()
    # mean = sum/count
    # sum = 0
    # for batch in dataloaders["trn_featureloader"](rng=None):
    #     logitsB = jnp.where(
    #         batch["marker"][..., None], logitsB, jnp.zeros_like(logitsB))
    #     logitsA = jnp.where(
    #         batch["marker"][..., None], logitsA, jnp.zeros_like(logitsA))
    #     sum += jnp.sum((logitsB-mean)**2, axis=[0, 1])
    #     sum += jnp.sum((logitsA-mean)**2, axis=[0, 1])
    # var = sum/count
    # std = jnp.sqrt(var)
    # print("mean", mean)
    # print("std", std)

    return dataloaders


class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale

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

    # equation (11) in I2SB
    def forward(self, x0, x1, training=True, **kwargs):
        # rng
        rng = kwargs["rng"]
        t_rng, n_rng = jax.random.split(rng, 2)

        _ts = jax.random.randint(t_rng, (x0.shape[0],), 1, self.n_T)  # (B,)
        sigma_weight_t = self.sigma_weight_t[_ts][:, None]  # (B, 1)
        sigma_t = self.sigma_t[_ts]
        mu_t = (sigma_weight_t*x0+(1-sigma_weight_t)*x1)
        bigsigma_t = self.bigsigma_t[_ts][:, None]  # (B, 1)

        # q(X_t|X_0,X_1) = N(X_t;mu_t,bigsigma_t)
        noise = jax.random.normal(n_rng, mu_t.shape)  # (B, d)
        x_t = mu_t + bigsigma_t*noise

        kwargs["t"] = _ts/self.n_T  # (B,)
        kwargs["training"] = training
        return x_t, sigma_t, kwargs

    def sample(self, rng, apply, x0, stats):
        shape = x0.shape
        batch_size = shape[0]
        x_n = x0  # (B, d)

        def body_fn(n, val):
            rng, x_n = val
            idx = self.n_T - n
            _rng, rng = jax.random.split(rng)
            t_n = jnp.array([idx/self.n_T])  # (1,)
            t_n = jnp.tile(t_n, [batch_size])  # (B,)

            h = jnp.where(idx > 1, jax.random.normal(
                _rng, shape), jnp.zeros(shape))  # (B, d)
            eps = apply(x_n, t_n)  # (2*B, d)
            x_n = (
                stats["oneover_sqrta"][idx] *
                (x_n-eps*stats["mab_over_sqrtmab"][idx])
                + stats["sqrt_beta_t"][idx]*h
            )  # (B, d)
            return rng, x_n

        _, x_n = jax.lax.fori_loop(0, self.n_T, body_fn, (rng, x_n))

        return x_n


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    init_rng, rng = jax.random.split(rng)

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_featureloaders(config)
    config.n_classes = dataloaders["num_classes"]

    beta1 = config.beta1
    beta2 = config.beta2
    dsb_stats = dsb_schedules(beta1, beta2, config.T)

    score_func = MLP(
        x_dim=10,
        pos_dim=16,
        encoder_layers=[16],
        decoder_layers=[128, 128]
    )

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)
        return init({'params': key},
                    x=jnp.ones((1, config.n_classes), model_dtype),
                    t=jnp.ones((1,)))
    variables = initialize_model(init_rng, score_func)

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders["trn_steps_per_epoch"])
    optimizer = optax.adam(learning_rate=scheduler)

    # Train state of diffusion bridge
    state = TrainState.create(
        apply_fn=score_func.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables["batch_stats"],
        rng=init_rng,
        dynamic_scale=dynamic_scale,
        betas=(beta1, beta2),
        n_T=config.T,
        **dsb_stats)

    # objective
    def mse_loss(noise, output):
        assert len(output.shape) == 2
        loss = jnp.sum((noise-output)**2, axis=1)
        return loss

    # training
    def step_train(state, batch, config):
        logitsB = batch["images"]  # mixture of other modes
        logitsA = batch["labels"]  # the current mode
        logitsB = normalize_logits(logitsB)
        logitsA = normalize_logits(logitsA)

        def loss_fn(params):
            logits_t, sigma_t, kwargs = state.forward(
                logitsA, logitsB, training=True, rng=state.rng)
            epsilon, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats}, logits_t, mutable="batch_stats", **kwargs)
            diff = (logits_t - logitsA) / sigma_t[:, None]
            loss = mse_loss(epsilon, diff)
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
        logitsB = batch["images"]
        logitsA = batch["labels"]
        logitsB = normalize_logits(logitsB)
        logitsA = normalize_logits(logitsA)
        logits_t, sigma_t, kwargs = state.forward(
            logitsA, logitsB, training=False, rng=state.rng)
        output = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats}, logits_t, **kwargs)
        diff = (logits_t - logitsA) / sigma_t[:, None]
        loss = mse_loss(output, diff)
        loss = jnp.sum(jnp.where(batch["marker"], loss, jnp.zeros_like(
            loss)))/jnp.sum(batch["marker"])
        count = jnp.sum(batch["marker"])
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))

        metrics = OrderedDict({"loss": loss, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        def apply(x_n, t_n):
            output = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                x=x_n, t=t_n, training=False)
            return output
        logitsB = batch["images"]  # current mode
        logitsA = batch["labels"]  # other modes
        logitsB = normalize_logits(logitsB)
        logitsA = normalize_logits(logitsA)
        f_gen = state.sample(
            state.rng, apply, logitsB, dsb_stats)
        f_gen = unnormalize_logits(f_gen)
        f_real = unnormalize_logits(logitsA)
        f_init = unnormalize_logits(logitsB)

        f_all = jnp.stack([f_init, f_gen, f_real], axis=-1)

        return f_all

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
        train_loader = dataloaders["featureloader"](rng=data_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        for batch_idx, batch in enumerate(train_loader, start=1):
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
            loss_rng, rng = jax.random.split(rng)
            state.replace(rng=loss_rng)
            metrics = p_step_valid(state, batch)
            valid_metrics.append(metrics)

            if batch_idx == 1:
                z_all = p_step_sample(state, batch)
                image_array = z_all[0][0]
                image_array = jax.nn.sigmoid(image_array)
                image_array = np.array(image_array)
                image_array = pixelize(image_array)
                image = Image.fromarray(image_array)
                image.save(f"dsb.png")
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
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=0.02, type=float)

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
