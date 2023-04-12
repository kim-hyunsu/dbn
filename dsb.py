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
from models.ddpm import MLP, visualize_dsb_stats
from models.bridge import FeatureUnet, dsb_schedules
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import pixelize, normalize_logits, unnormalize_logits, jprint

import matplotlib.pyplot as plt

import tensorflow as tf

def build_featureloaders(config):
    dir = config.features_dir
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
                with open(f"{dir}/train_features_M{mode_idx}S{i}.npy", "rb") as f:
                    train_logits = np.load(f)
                with open(f"{dir}/valid_features_M{mode_idx}S{i}.npy", "rb") as f:
                    valid_logits = np.load(f)
                with open(f"{dir}/test_features_M{mode_idx}S{i}.npy", "rb") as f:
                    test_logits = np.load(f)
                train_Blogits_list.append(train_logits)
                valid_Blogits_list.append(valid_logits)
                test_Blogits_list.append(test_logits)
        else:  # p_A (mixture of modes)
            for i in tqdm(range(n_samples_each_Amode)):
                with open(f"{dir}/train_features_M{mode_idx}S{i}.npy", "rb") as f:
                    train_logits = np.load(f)
                with open(f"{dir}/valid_features_M{mode_idx}S{i}.npy", "rb") as f:
                    valid_logits = np.load(f)
                with open(f"{dir}/test_features_M{mode_idx}S{i}.npy", "rb") as f:
                    test_logits = np.load(f)
                train_Alogits_list.append(train_logits)
                valid_Alogits_list.append(valid_logits)
                test_Alogits_list.append(test_logits)

    shift = 0

    train_logitsA = np.concatenate(train_Alogits_list, axis=0)
    train_logitsA = jnp.array(train_logitsA) + shift
    del train_Alogits_list

    train_logitsB = np.concatenate(train_Blogits_list, axis=0)
    train_logitsB = jnp.array(train_logitsB)
    del train_Blogits_list

    valid_logitsA = np.concatenate(valid_Alogits_list, axis=0)
    valid_logitsA = jnp.array(valid_logitsA) + shift
    del valid_Alogits_list

    valid_logitsB = np.concatenate(valid_Blogits_list, axis=0)
    valid_logitsB = jnp.array(valid_logitsB)
    del valid_Blogits_list

    test_logitsA = np.concatenate(test_Alogits_list, axis=0)
    test_logitsA = jnp.array(test_logitsA) + shift
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
    normalize = partial(normalize_logits,  features_dir=dir)
    unnormalize = partial(unnormalize_logits,  features_dir=dir)

    if config.compute_statistics:
        # >------------------------------------------------------------------
        # > Compute statistics (mean, std)
        # >------------------------------------------------------------------
        count = 0
        sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = batch["images"]
            logitsA = batch["labels"]
            logitsB = jnp.where(
                batch["marker"][..., None], logitsB, jnp.zeros_like(logitsB))
            logitsA = jnp.where(
                batch["marker"][..., None], logitsA, jnp.zeros_like(logitsA))
            sum += jnp.sum(logitsB, [0, 1])+jnp.sum(logitsA, [0, 1])
            count += 2*batch["marker"].sum()
        mean = sum/count
        sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = jnp.where(
                batch["marker"][..., None], logitsB, jnp.zeros_like(logitsB))
            logitsA = jnp.where(
                batch["marker"][..., None], logitsA, jnp.zeros_like(logitsA))
            sum += jnp.sum((logitsB-mean)**2, axis=[0, 1])
            sum += jnp.sum((logitsA-mean)**2, axis=[0, 1])
        var = sum/count
        std = jnp.sqrt(var)
        print("dims", len(mean), flush=True)
        print("mean", mean, flush=True)
        print("std", std, flush=True)

    return dataloaders, normalize, unnormalize


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
    alpos_t: Any
    alpos_weight_t: Any
    sigma_t_square: Any

    # equation (11) in I2SB
    def forward(self, x0, x1, training=True, **kwargs):
        """
            Args:
              x0: 
              x1:
              training (bool): True if training, False if evaluation
              **kwargs:
                keyword arguments

            Return:
              x_t:
                perturbed data
              sigma_t:
                sqrt(int_0^t (beta_t) dt), noise exponent integrated from 0 to t
              kwargs:
                keyword arguments
        """
        # rng
        rng = kwargs["rng"]
        t_rng, n_rng = jax.random.split(rng, 2)

        _ts = jax.random.randint(t_rng, (x0.shape[0],), 1, self.n_T)  # (B,)
        sigma_weight_t = self.sigma_weight_t[_ts][:, None]  # (B, 1)
        sigma_t = self.sigma_t[_ts]
        mu_t = sigma_weight_t * x0 + (1 - sigma_weight_t) * x1
        bigsigma_t = self.bigsigma_t[_ts][:, None]  # (B, 1)

        # q(X_t|X_0,X_1) = N(X_t;mu_t,bigsigma_t)
        noise = jax.random.normal(n_rng, mu_t.shape)  # (B, d)
        x_t = mu_t + jnp.sqrt(bigsigma_t) * noise

        kwargs["t"] = _ts/self.n_T  # (B,)
        kwargs["training"] = training
        return x_t, sigma_t, kwargs

    def sample(self, rng, apply, size, x0, stats, guide_w=0.,):
        n_samples = x0.shape[0]  # B
        new_size = (n_samples, *size)  # (B, d)
        x_n = x0  # (B, d)

        def body_fn(n, val):
            rng, x_n = val
            idx = self.n_T - n
            _rng, rng = jax.random.split(rng)
            t_n = jnp.array([idx/self.n_T])  # (1,)
            t_n = jnp.tile(t_n, [n_samples])  # (B,)

            x_n = jnp.tile(x_n, [2, 1])  # (2*B, d)
            t_n = jnp.tile(t_n, [2])  # (2*B,)

            h = jnp.where(idx > 1, jax.random.normal(
                _rng, new_size), jnp.zeros(new_size))  # (B, d)
            eps = apply(x_n, t_n)  # (2*B, d)
            eps1 = eps[:n_samples]  # (B, d)
            eps2 = eps[n_samples:]  # (B, d)
            eps = (1-guide_w)*eps1 + guide_w*eps2  # (B, d)
            x_n = x_n[:n_samples]  # (B, d)
            # x_n = (
            #     stats["oneover_sqrta"][idx] * (x_n - eps * stats["mab_over_sqrtmab"][idx]) +
            #     stats["sqrt_beta_t"][idx]   * h
            # )  # (B, d)
            x_0_eps = x_n - stats["sigma_t"][idx] * eps
            x_n = (
              stats["alpos_weight_t"][idx] * x_0_eps +
              (1 - stats["alpos_weight_t"][idx]) * x_n +
              jnp.sqrt(stats["alpos_weight_t"][idx] * stats["sigma_t_square"][idx]) * h
            )
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
    dataloaders, normalize_logits, unnormalize_logits = build_featureloaders(config)
    config.n_classes = dataloaders["num_classes"]
    config.ws_test = [0.0, 0.5, 2.0]

    # beta1 = 1e-4
    # beta2 = 0.02
    beta1 = config.beta1
    beta2 = config.beta2
    dsb_stats = dsb_schedules(beta1, beta2, config.T)

    if config.dataset == 'cifar10_logit':
      x_dim = 10
    elif config.dataset == 'cifar10_feature':
      x_dim = 64
    else:
      raise NotImplementedError()

    if config.network == 'mlp':
      score_func = MLP(
          x_dim=x_dim,
          pos_dim=128,
          encoder_layers=[256, 256, 256, 256, 256],
          decoder_layers=[256, 256, 256, 256, 256]
      )
    elif config.network == 'resnet':
      score_func = FeatureUnet(
          in_channels=x_dim,
          ver=config.version,
          n_feat=config.n_feat,
      )

    # Visualize all DSB stats
    visualize_dsb_stats(dsb_stats, config)

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args, **kwargs):
            return model.init(*args, **kwargs)
        return init({'params': key},
                    x=jnp.ones((1, x_dim), model_dtype),
                    t=jnp.ones((1,)))
    variables = initialize_model(init_rng, score_func)

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    if config.lr_schedule == 'cosine':
      scheduler = optax.cosine_decay_schedule(
          init_value=config.optim_lr,
          decay_steps=config.optim_ne * dataloaders["trn_steps_per_epoch"])
    elif config.lr_schedule == 'constant':
      scheduler = optax.constant_schedule(
        value=config.optim_lr
      )
    else:
      raise NotImplementedError()
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
            # I2SB, Eq. 12
            logits_t, sigma_t, kwargs = state.forward(
                logitsA, logitsB, training=True, rng=state.rng)
            epsilon, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats}, logits_t, mutable="batch_stats", **kwargs)
            diff = (logits_t - logitsA) / sigma_t[:, None] # (Xt-X0) / sigma_t
            loss = mse_loss(epsilon, diff) # ||eps(Xt,t;theta) - diff||_2^2
            # relative_loss = loss / diff
            loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
            count = jnp.sum(batch["marker"])
            loss = jnp.sum(loss) / count
            

            metrics = OrderedDict({
              "loss": loss,
              # "rel_loss": relative_loss
            })

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

        # relative_loss = 

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
        f_list = []
        for w_i, w in enumerate(config.ws_test):
            f_gen = state.sample(
                state.rng, apply, (x_dim,), logitsB, dsb_stats, guide_w=w)
            f_gen = unnormalize_logits(f_gen)
            f_real = unnormalize_logits(logitsA)
            f_init = unnormalize_logits(logitsB)

            f_all = jnp.stack([f_init, f_gen, f_real], axis=-1)
            f_list.append(f_all)

        return f_list

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_train = jax.pmap(
        partial(step_train, config=config), axis_name="batch")
    p_step_valid = jax.pmap(step_valid, axis_name="batch")
    p_step_sample = jax.pmap(
        partial(step_sample, config=config), axis_name="batch")
    state = jax_utils.replicate(state)
    best_loss = float("inf")

    # proj_coeff_list and orth_coeff_list
    proj_coeff_list, orth_coeff_list = [], []
    
    for epoch_idx, _ in enumerate(range(config.optim_ne), start=1):
        log_str = '[Epoch {:5d}/{:5d}] '.format(epoch_idx, config.optim_ne)
        data_rng, rng = jax.random.split(rng)

        train_metrics = []
        train_loader = dataloaders["featureloader"](rng=data_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        for batch_idx, batch in enumerate(train_loader, start=1):
            # TODO arange feature and input seperately
            loss_rng, rng = jax.random.split(rng)
            state = state.replace(rng=jax_utils.replicate(loss_rng))
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
            state = state.replace(rng=jax_utils.replicate(loss_rng))
            metrics = p_step_valid(state, batch)
            valid_metrics.append(metrics)

            if batch_idx == 1:
                z_list = p_step_sample(state, batch)
                for i, z_all in enumerate(z_list):
                    image_array = z_all[0][0]
                    image_array_float = image_array
                    image_array = jax.nn.sigmoid(image_array)
                    image_array = np.array(image_array)
                    image_array = pixelize(image_array)
                    image = Image.fromarray(image_array)
                    image.save(os.path.join(config.logdir, f"dsb_w{config.ws_test[i]}.png"))
                    if i == 0:
                      np.set_printoptions(precision=3)
                      print_fn(f"Epoch {epoch_idx}")
                      print_fn(f"Original  {image_array_float[:, 0]}")
                      print_fn(f"Generated {image_array_float[:, 1]}")
                      print_fn(f"Target    {image_array_float[:, 2]}")
                
                # How far is the generated samples are from the source and targets?
                z_arr = jnp.asarray(z_list)
                z_arr = jnp.reshape(z_arr, [3 * 8 * 64, x_dim, 3])
                src_to_tgt = z_arr[:, :, 2] - z_arr[:, :, 0]
                gen_to_tgt = z_arr[:, :, 2] - z_arr[:, :, 1]
                # projection mean: if zero, the generated vector moved to the direction of target vector from the source vector.
                #                  if higher, moved less / if lower (below zero), moved more.
                proj_coeff = jnp.sum(src_to_tgt * gen_to_tgt, axis=1) / jnp.sum(src_to_tgt * src_to_tgt, axis=1)
                proj_vec = proj_coeff[:, None] * gen_to_tgt
                proj_mean = jnp.mean(proj_coeff)
                # orthogonal mean: if zero, the generated vector is completely in the interpolation line.
                #                  if higher, the generated vector is apart from this.
                orth_vec = src_to_tgt - proj_vec
                orth_coeff = jnp.linalg.norm(orth_vec, axis=1) / jnp.linalg.norm(src_to_tgt, axis=1)
                orth_mean = jnp.mean(orth_coeff)

                proj_coeff_list.append(proj_mean)
                orth_coeff_list.append(orth_mean)

        # Plot projetive/orthogonal coefficients
        epoch_list = np.arange(1, epoch_idx + 1, dtype=float)
        
        figure_path = config.logdir
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set(xlim = [0., epoch_idx])
        ax1.set_title('Projective/Orthogonal coefficients')
        ax1.plot(epoch_list, np.squeeze(np.asarray(proj_coeff_list)), color='b', label='proj')
        ax1.plot(epoch_list, np.squeeze(np.asarray(orth_coeff_list)), color='r', label='orth')
        ax1.legend()
        fig1.savefig(os.path.join(figure_path,'projective_coefficients.png'))
        fig1.savefig(os.path.join(figure_path,'projective_coefficients.pdf'))


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

    parser.add_argument("--logdir", default='eval', type=str,
                        help='Directory of saved logs and figures.')
    parser.add_argument("--lr_schedule", default='cosine', type=str,
                        choices=['cosine', 'constant'])
    
    parser.add_argument("--beta1", default=1.5e-4, type=float,
                        help='beta1 (minimum beta) for DSB')
    parser.add_argument("--beta2", default=1.5e-3, type=float,
                        help='beta2 (maximum beta) for DSB')

    parser.add_argument("--dataset", default="cifar10_logit", type=str,
                        help='name of dataset')
    parser.add_argument("--method", default="i2sb", type=str,
                        help='name of diffusion-based method, default is I2SB.')
    parser.add_argument("--compute_statistics", default=False, type=bool,
                        help='Compute feature statistics.')
    parser.add_argument("--features_dir", default="features_last", type=str)
    parser.add_argument("--version", default="v1", type=str)
    parser.add_argument("--network", default="resnet", type=str)

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

    tf.io.gfile.mkdir(os.path.join(args.logdir))

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

    # verify dataset list
    dataset_list = [
        "cifar10_logit",
        "cifar10_feature"
    ]
    assert args.dataset in dataset_list, f"{args.dataset} is not a valid dataset."
    log_str = f'Using dataset {args.dataset}'
    print_fn(log_str)

    method_list = [
        "i2sb",
        "cfm"
    ]
    assert args.method in method_list, f"{args.method} is not a valid method."
    log_str = f"Using diffusion-based methoud {args.method}"
    print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
