from functools import partial
import os
import math
import orbax

from typing import Any, OrderedDict, Tuple

import flax
from flax.training import train_state, common_utils, checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.core.frozen_dict import freeze
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax
import jaxlib

import datetime
import wandb

import defaults_dsb as defaults
from tabulate import tabulate
import sys
from giung2.data.build import build_dataloaders, _build_dataloader
from giung2.metrics import evaluate_acc, evaluate_nll
from models.bridge import FeatureUnet, dsb_schedules, MLP
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from models.resnet import FlaxResNetClassifier
from utils import pixelize, normalize_logits, unnormalize_logits, jprint, model_list, logit_dir_list, feature_dir_list
from tqdm import tqdm


def build_featureloaders(config):
    dir = config.features_dir
    # B: current mode, A: other modes
    n_Amodes = config.n_Amodes
    n_samples_each_mode = config.n_samples_each_mode
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

    # classifying labels
    trn_labels = np.load(os.path.join(
        config.data_root, f'{config.data_name}/train_labels.npy'))
    tst_labels = np.load(os.path.join(
        config.data_root, f'{config.data_name}/test_labels.npy'))
    if config.data_name == "CIFAR10_x32":
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        num_classes = 10
    elif config.data_name == "CIFAR100_x32":
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        num_classes = 100

    trn_labels = jnp.tile(trn_labels, [n_samples_each_Bmode])
    val_labels = jnp.tile(val_labels, [n_samples_each_Bmode])
    tst_labels = jnp.tile(tst_labels, [n_samples_each_Bmode])

    train_logitsA = jnp.concatenate([
        train_logitsA,
        trn_labels[:, None]
    ], axis=-1)
    valid_logitsA = jnp.concatenate([
        valid_logitsA,
        val_labels[:, None]
    ], axis=-1)
    test_logitsA = jnp.concatenate([
        test_logitsA,
        tst_labels[:, None]
    ], axis=-1)

    dataloaders = dict(
        train_length=len(train_logitsA),
        valid_length=len(valid_logitsA),
        test_length=len(test_logitsA),
        num_classes=num_classes,
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
    if config.get_stats:
        # >------------------------------------------------------------------
        # > Compute statistics (mean, std)
        # >------------------------------------------------------------------
        count = 0
        sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = batch["images"]
            logitsA, labels = jnp.split(
                batch["labels"], [logitsB.shape[-1]], axis=-1)
            logitsB = jnp.where(
                batch["marker"][..., None], logitsB, jnp.zeros_like(logitsB))
            logitsA = jnp.where(
                batch["marker"][..., None], logitsA, jnp.zeros_like(logitsA))
            sum += jnp.sum(logitsB, [0, 1])+jnp.sum(logitsA, [0, 1])
            count += 2*batch["marker"].sum()
        mean = sum/count
        sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = batch["images"]
            logitsA, labels = jnp.split(
                batch["labels"], [logitsB.shape[-1]], axis=-1)
            mseB = jnp.where(
                batch["marker"][..., None], (logitsB-mean)**2, jnp.zeros_like(logitsB))
            mseA = jnp.where(
                batch["marker"][..., None], (logitsA-mean)**2, jnp.zeros_like(logitsA))
            sum += jnp.sum(mseB, axis=[0, 1])
            sum += jnp.sum(mseA, axis=[0, 1])
        var = sum/count
        std = jnp.sqrt(var)
        print("dims", len(mean))
        print("mean", mean)
        print("std", std)
        assert False, "Terminate Program"

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
        x_t = mu_t + noise*jnp.sqrt(bigsigma_t)

        kwargs["t"] = _ts/self.n_T  # (B,)
        kwargs["training"] = training
        return x_t, sigma_t, kwargs

    def sample(self, rng, apply, x0, stats):
        shape = x0.shape
        batch_size = shape[0]
        x_n = x0  # (B, d)

        def body_fn(n, val):
            rng, x_n, dir = val
            idx = self.n_T - n
            rng = jax.random.fold_in(rng, idx)
            t_n = jnp.array([idx/self.n_T])  # (1,)
            t_n = jnp.tile(t_n, [batch_size])  # (B,)

            h = jnp.where(idx > 1, jax.random.normal(
                rng, shape), jnp.zeros(shape))  # (B, d)
            eps = apply(x_n, t_n)  # (2*B, d)

            sigma_t = stats["sigma_t"][idx]
            alpos_weight_t = stats["alpos_weight_t"][idx]
            sigma_t_square = stats["sigma_t_square"][idx]
            std = jnp.sqrt(alpos_weight_t*sigma_t_square)

            x_0_eps = x_n - sigma_t*eps
            mean = alpos_weight_t*x_0_eps + (1-alpos_weight_t)*x_n
            x_n = mean + std * h  # (B, d)

            return rng, x_n, dir

        dir = jnp.zeros_like(x_n, dtype=bool)
        _, x_n, _ = jax.lax.fori_loop(0, self.n_T, body_fn, (rng, x_n, dir))

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
    dataloaders, normalize, unnormalize = build_featureloaders(config)
    config.n_classes = dataloaders["num_classes"]

    beta1 = config.beta1
    beta2 = config.beta2
    dsb_stats = dsb_schedules(beta1, beta2, config.T)
    # for k, v in dsb_stats.items():
    #     print(f"{k}: {v}")

    if config.features_dir in logit_dir_list:
        x_dim = 10
    elif config.features_dir in feature_dir_list:
        x_dim = 64

    score_func = FeatureUnet(
        in_channels=x_dim,
        ver=config.version,
        n_feat=config.n_feat,
    )

    classifier = FlaxResNetClassifier(
        num_classes=dataloaders["num_classes"]
    )

    # initialize model
    def initialize_model(key, model):
        return model.init(
            {'params': key},
            x=jnp.empty((1, x_dim), model_dtype),
            t=jnp.empty((1,))
        )
    variables = initialize_model(init_rng, score_func)
    variables_clsB = initialize_model(init_rng, classifier)
    variables_clsA = initialize_model(init_rng, classifier)

    def load_classifier(variables_cls, ckpt_dir):
        variables_cls = variables_cls.unfreeze()
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.join(ckpt_dir, "sghmc"),
            target=None
        )
        variables_cls["params"]["Dense_0"] = ckpt["model"]["params"]["Dense_0"]
        variables_cls = freeze(variables_cls)
        del ckpt

        return variables_cls

    variables_clsB = load_classifier(
        variables_clsB, model_list[0])  # current mode
    variables_clsA = load_classifier(
        variables_clsA, model_list[1])  # target mode

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
        batch_stats=variables.get("batch_stats"),
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

    def accuracy(f_gen, labels, marker, mode="A"):
        if f_gen.shape[-1] == 10:
            logits = f_gen
        else:
            if mode.lower() == "a":
                variables_cls_params = variables_clsA["params"]
            elif mode.lower() == "b":
                variables_cls_params = variables_clsB["params"]
            logits = classifier.apply(
                {"params": variables_cls_params}, f_gen
            )
        predictions = jax.nn.log_softmax(logits, axis=-1)
        acc = evaluate_acc(
            predictions, labels, log_input=True, reduction="none"
        )
        acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
        return acc

    def ensemble_accuracy(f_list, labels, marker, mode):
        assert len(f_list) == len(mode)
        avg_logits = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            if f_gen.shape[-1] == 10:
                logits = f_gen
            else:
                if m.lower() == "a":
                    variables_cls_params = variables_clsA["params"]
                elif m.lower() == "b":
                    variables_cls_params = variables_clsB["params"]
                logits = classifier.apply(
                    {"params": variables_cls_params}, f_gen
                )

            avg_logits += logits

            predictions = jax.nn.log_softmax(logits, axis=-1)
            acc = evaluate_acc(
                predictions, labels, log_input=True, reduction="none")
            nll = evaluate_nll(
                predictions, labels, log_input=True, reduction='none')
            acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
            nll = jnp.sum(jnp.where(marker, nll, jnp.zeros_like(nll)))
            each_acc.append((acc, nll))
        avg_logits /= len(f_list)

        predictions = jax.nn.log_softmax(avg_logits, axis=-1)
        acc = evaluate_acc(
            predictions, labels, log_input=True, reduction="none")
        nll = evaluate_nll(
            predictions, labels, log_input=True, reduction='none')
        acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(marker, nll, jnp.zeros_like(nll)))
        return (acc, nll), each_acc

    def cross_entropy(f_gen, labels, mode="A"):
        if f_gen.shape[-1] == 10:
            logits = f_gen
        else:
            if mode.lower() == "a":
                variables_cls_params = variables_clsA["params"]
            elif mode.lower() == "b":
                variables_cls_params = variables_clsB["params"]
            logits = classifier.apply(
                {"params": variables_cls_params}, f_gen
            )
        target = common_utils.onehot(
            labels, num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(
            target * jax.nn.log_softmax(logits, axis=-1),
            axis=-1)      # [B,]
        return loss

    def kl_divergence(logit_tar, logit_ref, marker):
        """
        logit_tar ~ q(x)
        logit_ref ~ p(x)

        return KL(q||p)
        """
        logq = jax.nn.log_softmax(logit_tar, axis=-1)
        logp = jax.nn.log_softmax(logit_ref, axis=-1)
        q = jnp.exp(logq)
        integrand = q*(logq-logp)
        assert len(integrand.shape) == 2
        kld = jnp.sum(integrand, axis=-1)
        kld = jnp.where(marker, kld, 0)
        return jnp.sum(kld)

    # training

    def step_train(state, batch, config):
        logitsB = batch["images"]  # the current mode
        # mixture of other modes
        logitsA, labels = jnp.split(
            batch["labels"], [logitsB.shape[-1]], axis=-1)
        logitsB = normalize(logitsB)
        logitsA = normalize(logitsA)

        def loss_fn(params):
            logits_t, sigma_t, kwargs = state.forward(
                logitsA, logitsB, training=True, rng=state.rng)
            params_dict = dict(params=params)
            mutable = []
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")
            epsilon, new_model_state = state.apply_fn(
                params_dict, logits_t, mutable=mutable, **kwargs)
            diff = (logits_t - logitsA) / sigma_t[:, None]
            loss = mse_loss(epsilon, diff)
            celoss = cross_entropy(unnormalize(logits_t), labels)
            count = jnp.sum(batch["marker"])
            loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
            loss = jnp.sum(loss) / count
            celoss = jnp.where(batch["marker"], celoss, jnp.zeros_like(celoss))
            celoss = jnp.sum(celoss) / count

            totalloss = loss + config.gamma*celoss

            metrics = OrderedDict(
                {"loss": loss*count, "count": count, "celoss": celoss*count})

            return totalloss, (metrics, new_model_state)

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
            grads=grads, batch_stats=new_model_state.get('batch_stats'))
        metrics = jax.lax.psum(metrics, axis_name="batch")

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
        logitsA, _ = jnp.split(
            batch["labels"], [logitsB.shape[-1]], axis=-1)
        logitsB = normalize(logitsB)
        logitsA = normalize(logitsA)
        logits_t, sigma_t, kwargs = state.forward(
            logitsA, logitsB, training=False, rng=state.rng)
        params_dict = dict(params=state.params)
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
        output = state.apply_fn(params_dict, logits_t, **kwargs)
        diff = (logits_t - logitsA) / sigma_t[:, None]
        loss = mse_loss(output, diff)
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
        count = jnp.sum(batch["marker"])
        loss = jnp.where(count > 0, jnp.sum(loss)/count, 0)

        metrics = OrderedDict({"loss": loss*count, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        def apply(x_n, t_n):
            params_dict = dict(params=state.params)
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
            output = state.apply_fn(params_dict, x=x_n, t=t_n, training=False)
            return output
        _logitsB = batch["images"]  # current mode
        _logitsA, labels = jnp.split(
            batch["labels"], [_logitsB.shape[-1]], axis=-1)
        labels = jnp.squeeze(labels, -1)
        logitsB = normalize(_logitsB)
        logitsA = normalize(_logitsA)
        f_gen = state.sample(
            state.rng, apply, logitsB, dsb_stats)
        f_gen = unnormalize(f_gen)
        f_real = _logitsA
        f_init = _logitsB

        f_all = jnp.stack([f_init, f_gen, f_real], axis=-1)

        (ens_acc, ens_nll), ((b_acc, b_nll), (a_acc, a_nll)) = ensemble_accuracy(
            [f_init, f_gen], labels, batch["marker"], ["B", "A"])
        count = jnp.sum(batch["marker"])
        kld = kl_divergence(f_gen, f_real, batch["marker"])

        metrics = OrderedDict({
            "acc": a_acc, "nll": a_nll,
            "ens_acc": ens_acc, "ens_nll": ens_nll,
            "count": count, "kld": kld
        })
        metrics = jax.lax.psum(metrics, axis_name='batch')

        return f_all, metrics

    def step_acc_ref(state, batch):
        _logitsB = batch["images"]  # current mode
        _logitsA, labels = jnp.split(
            batch["labels"], [_logitsB.shape[-1]], axis=-1)
        labels = jnp.squeeze(labels, -1)
        f_real = _logitsA
        f_init = _logitsB
        (ens_acc, ens_nll), ((b_acc, b_nll), (a_acc, a_nll)) = ensemble_accuracy(
            [f_init, f_real], labels, batch["marker"], ["B", "A"])
        metrics = OrderedDict(
            {"acc_ref": a_acc, "nll_ref": a_nll,
             "ens_acc_ref": ens_acc, "ens_nll_ref": ens_nll})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    def save_samples(state, batch, epoch_idx, train):
        if (epoch_idx+1) % 50 == 0 and os.environ.get("DEBUG") != True:
            z_all, _ = p_step_sample(state, batch)
            image_array = z_all[0][0]
            image_array = jax.nn.sigmoid(image_array)
            image_array = np.array(image_array)
            image_array = pixelize(image_array)
            image = Image.fromarray(image_array)
            image.save(train_image_name if train else valid_image_name)

    def log_wandb(object):
        to_summary = [
            "trn/acc_ref", "trn/nll_ref", "trn/ens_acc_ref", "trn/ens_nll_ref",
            "val/acc_ref", "val/nll_ref", "val/ens_acc_ref", "val/ens_nll_ref",
            "tst/acc_ref", "tst/nll_ref", "tst/ens_acc_ref", "tst/ens_nll_ref",
            "tst/acc", "tst/nll",
            "tst/ens_acc", "tst/ens_nll",
            "tst/loss", "tst/kld", "tst/ens_nll",
        ]
        for k in to_summary:
            value = object.get(k)
            if value is None:
                continue
            wandb.run.summary[k] = value
            del object[k]
        wandb.log(object)

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_train = jax.pmap(
        partial(step_train, config=config), axis_name="batch")
    p_step_valid = jax.pmap(step_valid, axis_name="batch")
    p_step_sample = jax.pmap(
        partial(step_sample, config=config), axis_name="batch")
    p_step_acc_ref = jax.pmap(step_acc_ref, axis_name="batch")
    state = jax_utils.replicate(state)
    best_loss = float("inf")
    image_name = datetime.datetime.now().strftime('%y%m%d%H%M%S%f')
    feature_type = "last" if config.features_dir in feature_dir_list else "logit"
    train_image_name = f"images/{config.version}train{feature_type}{image_name}.png"
    valid_image_name = f"images/{config.version}valid{feature_type}{image_name}.png"
    trn_summary = dict()
    val_summary = dict()
    tst_summary = dict()

    run = wandb.init(
        project="dsb-bnn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/ens_acc", summary="max")
    wandb.define_metric("val/ens_nll", summary="min")

    for epoch_idx, _ in enumerate(tqdm(range(config.optim_ne)), start=1):
        rng = jax.random.fold_in(rng, epoch_idx)
        data_rng, rng = jax.random.split(rng)

        train_metrics = []
        train_loader = dataloaders["featureloader"](rng=data_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        for batch_idx, batch in enumerate(train_loader, start=1):
            rng = jax.random.fold_in(rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(rng))
            state, metrics = p_step_train(state, batch)

            if epoch_idx == 1:
                acc_ref_metrics = p_step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if config.show:
                if batch_idx == dataloaders["trn_steps_per_epoch"]:
                    save_samples(state, batch, epoch_idx, train=True)
            else:
                if (epoch_idx+1) % 1 == 0:
                    _, acc_metrics = p_step_sample(state, batch)
                    metrics.update(acc_metrics)

            train_metrics.append(metrics)

        train_metrics = common_utils.get_metrics(train_metrics)
        trn_summarized = {f'trn/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), train_metrics).items()}
        for k, v in trn_summarized.items():
            if "count" not in k and "lr" not in k:
                trn_summarized[k] /= trn_summarized["trn/count"]
        del trn_summarized["trn/count"]
        del trn_summarized["trn/celoss"]
        trn_summary.update(trn_summarized)
        log_wandb(trn_summary)

        if state.batch_stats is not None:
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        valid_metrics = []
        valid_loader = dataloaders["val_featureloader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        for batch_idx, batch in enumerate(valid_loader, start=1):
            rng = jax.random.fold_in(rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(rng))
            metrics = p_step_valid(state, batch)

            if epoch_idx == 1:
                acc_ref_metrics = p_step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if config.show:
                if batch_idx == dataloaders["val_steps_per_epoch"]:
                    save_samples(state, batch, epoch_idx, train=False)
            else:
                if (epoch_idx+1) % 1 == 0:
                    _, acc_metrics = p_step_sample(state, batch)
                    metrics.update(acc_metrics)
                    assert jnp.all(metrics["count"] == acc_metrics["count"])

            valid_metrics.append(metrics)

        valid_metrics = common_utils.get_metrics(valid_metrics)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), valid_metrics).items()}
        for k, v in val_summarized.items():
            if "count" not in k and "lr" not in k:
                val_summarized[k] /= val_summarized["val/count"]
        del val_summarized["val/count"]
        val_summary.update(val_summarized)
        log_wandb(val_summary)

        if best_loss > val_summarized["val/loss"]:
            test_metrics = []
            test_loader = dataloaders["tst_featureloader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            for batch_idx, batch in enumerate(test_loader, start=1):
                rng = jax.random.fold_in(rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(rng))
                _, metrics = p_step_sample(state, batch)
                if best_loss == float("inf"):
                    acc_ref_metrics = p_step_acc_ref(state, batch)
                    metrics.update(acc_ref_metrics)
                test_metrics.append(metrics)
            test_metrics = common_utils.get_metrics(test_metrics)
            tst_summarized = {
                f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), test_metrics).items()}
            for k, v in tst_summarized.items():
                if "count" not in k and "lr" not in k:
                    tst_summarized[k] /= tst_summarized["tst/count"]
            del tst_summarized["tst/count"]
            tst_summary.update(tst_summarized)
            log_wandb(tst_summary)
            best_loss = val_summarized['val/loss']

            if config.save:
                save_state = jax_utils.unreplicate(state)
                ckpt = dict(model=save_state, config=vars(
                    config), best_loss=best_loss)
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(ckpt_dir=config.save,
                                            target=ckpt,
                                            step=epoch_idx,
                                            overwrite=True,
                                            orbax_checkpointer=orbax_checkpointer)

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_ne', default=300, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=1e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=1e-4, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--T", default=50, type=int)
    parser.add_argument("--n_feat", default=64, type=int)
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=0.02, type=float)
    parser.add_argument("--features_dir", default="features_fixed", type=str)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--gamma", default=0., type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--get_stats", action="store_true")
    parser.add_argument("--n_Amodes", default=1, type=int)
    parser.add_argument("--n_samples_each_mode", default=1, type=int)

    args = parser.parse_args()

    if args.seed < 0:
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
