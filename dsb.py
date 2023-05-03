import time
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
from data.build import build_dataloaders, _build_dataloader, _build_featureloader
from giung2.metrics import evaluate_acc, evaluate_nll
from models.resnet import FlaxResNetClassifier, FlaxResNetClassifier2
from models.bridge import FeatureUnet, dsb_schedules, MLP
from models.i2sb import UNetModel
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import WandbLogger, pixelize, normalize_logits, unnormalize_logits
from utils import model_list, logit_dir_list, feature_dir_list, feature2_dir_list
from utils import get_info_in_dir, jprint, expand_to_broadcast
from tqdm import tqdm
from functools import partial


def build_featureloaders(config, rng=None):
    def load_arrays(dir, div, prop, mode_idx, i, length=None):
        path = f"{dir}/{div}_{prop}_M{mode_idx}S{i}.npy"
        if not os.path.exists(path):
            return jnp.ones((length,))
        with open(path, "rb") as f:
            logits = np.load(f)
        return logits

    def get_context_dir(dir):
        f_dir = dir.split("_")[0]
        f_dir = dir.replace(f_dir, f"{f_dir}_last")
        return f_dir

    dir = config.features_dir
    f_dir = get_context_dir(dir)
    if config.context:
        assert f_dir == config.contexts_dir
        f_dir = config.contexts_dir
    # B: current mode, A: other modes
    n_Amodes = config.n_Amodes
    n_samples_each_mode = config.n_samples_each_mode
    n_samples_each_Bmode = n_Amodes*n_samples_each_mode
    n_samples_each_Amode = n_samples_each_mode
    train_Alogits_list = []
    train_Blogits_list = []
    train_Afeatures_list = []
    train_Bfeatures_list = []
    train_lambdas_list = []
    train_lambdas_checker = []
    valid_Alogits_list = []
    valid_Blogits_list = []
    valid_Afeatures_list = []
    valid_Bfeatures_list = []
    valid_lambdas_list = []
    test_Alogits_list = []
    test_Blogits_list = []
    test_Afeatures_list = []
    test_Bfeatures_list = []
    test_lambdas_list = []
    for mode_idx in range(1+n_Amodes):  # total 1+n_Amodes
        if mode_idx == 0:  # p_B
            for i in tqdm(range(n_samples_each_Bmode)):
                # logits
                train_logits = load_arrays(
                    dir, "train", "features", mode_idx, i)
                valid_logits = load_arrays(
                    dir, "valid", "features", mode_idx, i)
                test_logits = load_arrays(
                    dir, "test", "features", mode_idx, i)
                train_Blogits_list.append(train_logits)
                valid_Blogits_list.append(valid_logits)
                test_Blogits_list.append(test_logits)
                # lambdas
                train_lambdas = load_arrays(
                    dir, "train", "lambdas", mode_idx, i, train_logits.shape[0])
                valid_lambdas = load_arrays(
                    dir, "valid", "lambdas", mode_idx, i, valid_logits.shape[0])
                test_lambdas = load_arrays(
                    dir, "test", "lambdas", mode_idx, i, test_logits.shape[0])
                train_lambdas_list.append(train_lambdas)
                valid_lambdas_list.append(valid_lambdas)
                test_lambdas_list.append(test_lambdas)
                # features (for context)
                train_features = load_arrays(
                    f_dir, "train", "features", mode_idx, i, train_logits.shape[0])
                valid_features = load_arrays(
                    f_dir, "valid", "features", mode_idx, i, valid_logits.shape[0])
                test_features = load_arrays(
                    f_dir, "test", "features", mode_idx, i, test_logits.shape[0])
                train_Bfeatures_list.append(train_features)
                valid_Bfeatures_list.append(valid_features)
                test_Bfeatures_list.append(test_features)
        else:  # p_A (mixture of modes)
            for i in tqdm(range(n_samples_each_Amode)):
                # logits
                train_logits = load_arrays(
                    dir, "train", "features", mode_idx, i)
                valid_logits = load_arrays(
                    dir, "valid", "features", mode_idx, i)
                test_logits = load_arrays(
                    dir, "test", "features", mode_idx, i)
                train_Alogits_list.append(train_logits)
                valid_Alogits_list.append(valid_logits)
                test_Alogits_list.append(test_logits)
                # lambdas (for sync)
                train_lambdas = load_arrays(
                    dir, "train", "lambdas", mode_idx, i, train_logits.shape[0])
                train_lambdas_checker.append(train_lambdas)
                # features (for context)
                train_features = load_arrays(
                    f_dir, "train", "features", mode_idx, i, train_logits.shape[0])
                valid_features = load_arrays(
                    f_dir, "valid", "features", mode_idx, i, valid_logits.shape[0])
                test_features = load_arrays(
                    f_dir, "test", "features", mode_idx, i, test_logits.shape[0])
                train_Afeatures_list.append(train_features)
                valid_Afeatures_list.append(valid_features)
                test_Afeatures_list.append(test_features)

    train_logitsA = np.concatenate(train_Alogits_list, axis=0)
    train_logitsA = jnp.array(train_logitsA)
    del train_Alogits_list

    train_logitsB = np.concatenate(train_Blogits_list, axis=0)
    train_logitsB = jnp.array(train_logitsB)
    del train_Blogits_list

    train_featuresA = np.concatenate(train_Afeatures_list, axis=0)
    train_featuresA = jnp.array(train_featuresA)
    del train_Afeatures_list

    train_featuresB = np.concatenate(train_Bfeatures_list, axis=0)
    train_featuresB = jnp.array(train_featuresB)
    del train_Bfeatures_list

    valid_logitsA = np.concatenate(valid_Alogits_list, axis=0)
    valid_logitsA = jnp.array(valid_logitsA)
    del valid_Alogits_list

    valid_logitsB = np.concatenate(valid_Blogits_list, axis=0)
    valid_logitsB = jnp.array(valid_logitsB)
    del valid_Blogits_list

    valid_featuresA = np.concatenate(valid_Afeatures_list, axis=0)
    valid_featuresA = jnp.array(valid_featuresA)
    del valid_Afeatures_list

    valid_featuresB = np.concatenate(valid_Bfeatures_list, axis=0)
    valid_featuresB = jnp.array(valid_featuresB)
    del valid_Bfeatures_list

    test_logitsA = np.concatenate(test_Alogits_list, axis=0)
    test_logitsA = jnp.array(test_logitsA)
    del test_Alogits_list

    test_logitsB = np.concatenate(test_Blogits_list, axis=0)
    test_logitsB = jnp.array(test_logitsB)
    del test_Blogits_list

    test_featuresA = np.concatenate(test_Afeatures_list, axis=0)
    test_featuresA = jnp.array(test_featuresA)
    del test_Afeatures_list

    test_featuresB = np.concatenate(test_Bfeatures_list, axis=0)
    test_featuresB = jnp.array(test_featuresB)
    del test_Bfeatures_list

    train_lambdas = np.concatenate(train_lambdas_list, axis=0)
    train_lambdas = jnp.array(train_lambdas)
    del train_lambdas_list

    train_lambdas_checker = np.concatenate(train_lambdas_checker, axis=0)
    train_lambdas_checker = jnp.array(train_lambdas_checker)
    assert jnp.all(train_lambdas_checker == train_lambdas)
    del train_lambdas_checker

    valid_lambdas = np.concatenate(valid_lambdas_list, axis=0)
    valid_lambdas = jnp.array(valid_lambdas)
    del valid_lambdas_list

    test_lambdas = np.concatenate(test_lambdas_list, axis=0)
    test_lambdas = jnp.array(test_lambdas)
    del test_lambdas_list

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

    _, repeats = get_info_in_dir(dir)
    trn_labels = jnp.tile(trn_labels, [n_samples_each_Bmode*repeats])
    val_labels = jnp.tile(val_labels, [n_samples_each_Bmode])
    tst_labels = jnp.tile(tst_labels, [n_samples_each_Bmode])

    # train_logitsB = jnp.concatenate([
    #     train_logitsB,
    #     train_lambdas[:, None]
    # ], axis=-1)
    # valid_logitsB = jnp.concatenate([
    #     valid_logitsB,
    #     valid_lambdas[:, None]
    # ], axis=-1)
    # test_logitsB = jnp.concatenate([
    #     test_logitsB,
    #     test_lambdas[:, None]
    # ], axis=-1)

    # train_logitsA = jnp.concatenate([
    #     train_logitsA,
    #     trn_labels[:, None]
    # ], axis=-1)
    # valid_logitsA = jnp.concatenate([
    #     valid_logitsA,
    #     val_labels[:, None]
    # ], axis=-1)
    # test_logitsA = jnp.concatenate([
    #     test_logitsA,
    #     tst_labels[:, None]
    # ], axis=-1)

    def mix_dataset(rng, train_logits, valid_logits):
        assert len(train_logits) == len(valid_logits)
        # train_indices = [sum(l.shape[-1] for l in train_logits[:i+1])
        #                  for i in range(len(train_logits))][:-1]
        # train_logits = jnp.concatenate(train_logits, axis=-1)
        # valid_indices = [sum(l.shape[-1] for l in valid_logits[:i+1])
        #                  for i in range(len(valid_logits))][:-1]
        # valid_logits = jnp.concatenate(valid_logits, axis=-1)
        # train_length = train_logits.shape[0]
        # logits = jnp.concatenate([train_logits, valid_logits], axis=0)
        # mixed_logits = jax.random.permutation(rng, logits, axis=0)
        # _train_logits = mixed_logits[:train_length]
        # _valid_logits = mixed_logits[train_length:]
        # train_logits = jnp.split(_train_logits, train_indices, axis=-1)
        # valid_logits = jnp.split(_valid_logits, valid_indices, axis=-1)
        train_size = train_logits[0].shape[0]
        valid_size = valid_logits[0].shape[0]
        perm_order = jax.random.permutation(
            rng, jnp.arange(train_size+valid_size))
        new_train_logits = []
        new_valid_logits = []
        for train, valid in zip(train_logits, valid_logits):
            assert train.shape[0] == train_size
            assert valid.shape[0] == valid_size
            total = jnp.concatenate([train, valid], axis=0)
            mixed_total = total[perm_order]
            new_train_logits.append(mixed_total[:train_size])
            new_valid_logits.append(mixed_total[train_size:])

        return new_train_logits, new_valid_logits

    if config.take_valid:
        (
            (
                train_logitsB,
                train_logitsA,
                train_featuresB,
                train_featuresA,
                trn_labels,
                train_lambdas
            ),
            (
                valid_logitsB,
                valid_logitsA,
                valid_featuresB,
                valid_featuresA,
                val_labels,
                valid_lambdas
            )
        ) = mix_dataset(
            rng,
            [
                train_logitsB,
                train_logitsA,
                train_featuresB,
                train_featuresA,
                trn_labels,
                train_lambdas
            ],
            [
                valid_logitsB,
                valid_logitsA,
                valid_featuresB,
                valid_featuresA,
                val_labels,
                valid_lambdas
            ]
        )

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
        # _build_dataloader,
        _build_featureloader,
        images=train_logitsB,
        labels=train_logitsA,
        cls_labels=trn_labels,
        lambdas=train_lambdas,
        contexts=train_featuresB,
        batch_size=config.optim_bs,
        shuffle=True,
        transform=None
    )

    dataloaders["trn_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=train_logitsB,
        labels=train_logitsA,
        cls_labels=trn_labels,
        lambdas=train_lambdas,
        contexts=train_featuresB,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["val_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=valid_logitsB,
        labels=valid_logitsA,
        cls_labels=val_labels,
        lambdas=valid_lambdas,
        contexts=valid_featuresB,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["tst_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=test_logitsB,
        labels=test_logitsA,
        cls_labels=tst_labels,
        lambdas=test_lambdas,
        contexts=test_featuresB,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    normalize = partial(normalize_logits,  features_dir=dir)
    unnormalize = partial(unnormalize_logits,  features_dir=dir)
    f_normalize = partial(
        normalize_logits,  features_dir=f_dir) if config.context else None
    f_unnormalize = partial(
        unnormalize_logits,  features_dir=f_dir) if config.context else None

    # def get_data(batch, key):
    #     array = batch[key]
    #     logits, scalar = jnp.split(
    #         array, [array.shape[-1]-1], axis=-1
    #     )
    #     logits = jnp.where(
    #         batch["marker"][..., None], logits, float("nan")*jnp.ones_like(logits))
    #     lambdas = jnp.squeeze(scalar, axis=-1)
    #     return logits, lambdas

    if config.get_stats:
        # ------------------------------------------------------------------
        #  Compute statistics (mean, std)
        # ------------------------------------------------------------------
        count = 0
        _sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            # logitsB, _ = get_data(batch, "images")
            # logitsA, _ = get_data(batch, "labels")
            logitsB = batch["images"]
            logitsA = batch["labels"]
            # 0: pmap dimension, 1: batch dimension
            _sum += jnp.sum(logitsB, [0, 1])+jnp.sum(logitsA, [0, 1])
            count += 2*batch["marker"].sum()
        mean = _sum/count
        _sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            # logitsB, _ = get_data(batch, "images")
            # logitsA, _ = get_data(batch, "labels")
            logitsB = batch["images"]
            logitsA = batch["labels"]
            marker = expand_to_broadcast(batch["marker"], logitsB, axis=2)
            mseB = jnp.where(marker, (logitsB-mean)**2,
                             jnp.zeros_like(logitsB))
            mseA = jnp.where(marker, (logitsA-mean)**2,
                             jnp.zeros_like(logitsA))
            _sum += jnp.sum(mseB, axis=[0, 1])
            _sum += jnp.sum(mseA, axis=[0, 1])
        var = _sum/count
        std = jnp.sqrt(var)
        print("dims", mean.shape)
        print("mean", mean)
        print("std", std)
        assert False, "Terminate Program"

    # return dataloaders, normalize, unnormalize, get_data, f_normalize, f_unnormalize
    return dataloaders, normalize, unnormalize, f_normalize, f_unnormalize


class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_params: Any

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
        # sigma_weight_t = self.sigma_weight_t[_ts][:, None]  # (B, 1)
        sigma_weight_t = self.sigma_weight_t[_ts]  # (B,)
        sigma_weight_t = expand_to_broadcast(sigma_weight_t, x0, axis=1)
        sigma_t = self.sigma_t[_ts]
        mu_t = (sigma_weight_t*x0+(1-sigma_weight_t)*x1)
        # bigsigma_t = self.bigsigma_t[_ts][:, None]  # (B, 1)
        bigsigma_t = self.bigsigma_t[_ts]  # (B,)
        bigsigma_t = expand_to_broadcast(bigsigma_t, mu_t, axis=1)

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
            rng, x_n = val
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

            return rng, x_n

        _, x_n = jax.lax.fori_loop(0, self.n_T, body_fn, (rng, x_n))

        return x_n


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    init_rng, data_rng, rng = jax.random.split(rng, 3)

    if config.features_dir in logit_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = 100
            ctx_dim = 128
        else:
            x_dim = 10
            ctx_dim = 64
    elif config.features_dir in feature_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = 128
            ctx_dim = (8, 8, 128)
        else:
            x_dim = 64
            ctx_dim = (8, 8, 64)
    elif config.features_dir in feature2_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = (8, 8, 128)
            ctx_dim = (0, 0, 0)  # TEMP
        else:
            x_dim = (8, 8, 64)
            ctx_dim = (0, 0, 0)  # TEMP
    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    (
        # dataloaders, normalize, unnormalize, get_data, f_normalize, f_unnormalize
        dataloaders, normalize, unnormalize, f_normalize, f_unnormalize
    ) = build_featureloaders(
        config, data_rng)
    config.n_classes = dataloaders["num_classes"]

    beta1 = config.beta1
    beta2 = config.beta2
    dsb_stats = dsb_schedules(beta1, beta2, config.T)

    if isinstance(x_dim, int):
        score_func = FeatureUnet(
            in_channels=x_dim,
            ver=config.version,
            n_feat=config.n_feat,
            droprate=config.droprate,
            context=ctx_dim if config.context else 0
        )
        classifier = FlaxResNetClassifier(
            num_classes=dataloaders["num_classes"]
        )
    elif isinstance(x_dim, tuple) and len(x_dim) > 1:
        image_size = x_dim[0]
        num_channels = config.n_feat
        channel_mult = ""
        learn_sigma = config.learn_sigma
        num_res_blocks = config.num_res_blocks
        class_cond = False
        attention_resolutions = "16"
        num_heads = config.num_heads
        num_head_channels = -1
        num_heads_upsample = -1
        use_scale_shift_norm = config.use_scale_shift_norm
        dropout = config.droprate
        resblock_updown = config.resblock_updown
        use_new_attention_order = config.use_new_attention_order
        in_channels = x_dim[-1]
        if channel_mult == "":
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 4)
            elif image_size == 16:
                channel_mult = (1, 2, 4)
            elif image_size == 8:
                channel_mult = (1, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(int(ch_mult)
                                 for ch_mult in channel_mult.split(","))

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        score_func = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else 2*in_channels),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(dataloaders["num_classes"] if class_cond else None),
            dtype=model_dtype,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            context=ctx_dim if config.context else None
        )
        classifier = FlaxResNetClassifier2(
            num_classes=dataloaders["num_classes"]
        )

    # initialize model

    def initialize_model(key, model):
        if isinstance(x_dim, int):
            init_x = jnp.empty((1, x_dim), model_dtype)
        else:
            init_x = jnp.empty((1, *x_dim), model_dtype)
        if isinstance(ctx_dim, int):
            init_ctx = jnp.empty((1, ctx_dim))
        else:
            init_ctx = jnp.empty((1, *ctx_dim))
        return model.init(
            {'params': key},
            x=init_x,
            t=jnp.empty((1,)),
            ctx=init_ctx,
            training=False
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
        params = ckpt["model"]["params"].get("Dense_0")
        if params is None:
            params = ckpt["model"]["params"]["head"]
        variables_cls["params"]["Dense_0"] = params
        variables_cls = freeze(variables_cls)
        del ckpt

        return variables_cls

    name = config.data_name
    style = config.model_style
    shared = config.shared_head
    variables_clsB = load_classifier(
        variables_clsB, model_list(name, style, shared)[0])  # current mode
    variables_clsA = load_classifier(
        variables_clsA, model_list(name, style, shared)[1])  # target mode

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders["trn_steps_per_epoch"])
    optimizer = optax.adam(learning_rate=scheduler)
    if isinstance(score_func, UNetModel):
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}
        param_partitions = freeze(flax.traverse_util.path_aware_map(
            lambda path, _: "frozen" if "frozen" in path else "trainable", variables["params"]))
        optimizer = optax.multi_transform(
            partition_optimizer, param_partitions)

    # Train state of diffusion bridge
    state = TrainState.create(
        apply_fn=score_func.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=init_rng,
        dynamic_scale=dynamic_scale,
        betas=(beta1, beta2),
        n_T=config.T,
        **dsb_stats)

    # objective
    def mse_loss(noise, output):
        # assert len(output.shape) == 2
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum((noise-output)**2, axis=sum_axis)
        return loss

    def get_logits(feat, mode):
        if mode.lower() == "a":
            variables_cls_params = variables_clsA["params"]
        elif mode.lower() == "b":
            variables_cls_params = variables_clsB["params"]
        logits = classifier.apply(
            {"params": variables_cls_params}, feat
        )
        return logits

    def ensemble_accuracy(f_list, labels, marker, mode):
        assert len(f_list) == len(mode)
        avg_logits = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            if "last" in config.features_dir:  # last or last2
                logits = get_logits(f_gen, m)
            else:
                logits = f_gen

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
        if "last" in config.features_dir:  # last or last2
            logits = get_logits(f_gen, mode)
        else:
            logits = f_gen
        target = common_utils.onehot(
            labels, num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(
            target * jax.nn.log_softmax(logits, axis=-1),
            axis=-1)      # [B,]
        return loss

    def kl_divergence(logit_tar, logit_ref, marker, mode_tar="A", mode_ref="A"):
        """
        logit_tar ~ q(x)
        logit_ref ~ p(x)

        return KL(q||p)
        """

        if "last" in config.features_dir:
            logit_tar = get_logits(logit_tar, mode_tar)
            logit_ref = get_logits(logit_ref, mode_ref)
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
        # logitsB, lambdas = get_data(batch, "images")# the current mode
        # logitsA, labels = get_data(batch, "labels")# mixture of other modes
        logitsB = batch["images"]  # the current mode
        logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        logitsB = normalize(logitsB)
        logitsA = normalize(logitsA)
        contexts = f_normalize(batch["contexts"]) if config.context else None

        def loss_fn(params):
            logits_t, sigma_t, kwargs = state.forward(
                logitsA, logitsB, training=True, rng=state.rng)
            params_dict = dict(params=params)
            rngs_dict = dict(dropout=state.rng)
            mutable = []
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")
            epsilon, new_model_state = state.apply_fn(
                params_dict, logits_t, ctx=contexts, mutable=mutable, rngs=rngs_dict, **kwargs)
            _sigma_t = expand_to_broadcast(sigma_t, logits_t, axis=1)
            diff = (logits_t - logitsA) / _sigma_t
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
        a = config.ema_decay
        new_state = new_state.replace(
            ema_params=jax.tree_util.tree_map(
                lambda wt, ema_tm1: jnp.where(a < 1, a*wt + (1-a)*ema_tm1, wt),
                new_state.params,
                new_state.ema_params))
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
        # normalize
        logitsB = batch["images"]  # the current mode
        logitsA = batch["labels"]  # mixture of other modes
        contexts = batch["contexts"]
        logitsB = normalize(logitsB)
        logitsA = normalize(logitsA)
        contexts = f_normalize(contexts) if config.context else None
        # get interpolation
        logits_t, sigma_t, kwargs = state.forward(
            logitsA, logitsB, training=False, rng=state.rng)
        # get epsilon
        params_dict = dict(params=state.ema_params)
        rngs_dict = dict(dropout=state.rng)
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
        output = state.apply_fn(
            params_dict, logits_t, ctx=contexts, rngs=rngs_dict, **kwargs)
        # compute loss
        # diff = (logits_t - logitsA) / sigma_t[:, None]
        _sigma_t = expand_to_broadcast(sigma_t, logits_t, axis=1)
        diff = (logits_t - logitsA) / _sigma_t
        loss = mse_loss(output, diff)
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
        count = jnp.sum(batch["marker"])
        loss = jnp.where(count > 0, jnp.sum(loss)/count, 0)
        # collect metrics
        metrics = OrderedDict({"loss": loss*count, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        context = f_normalize(batch["contexts"]) if config.context else None

        def apply(x_n, t_n):
            params_dict = dict(params=state.ema_params)
            rngs_dict = dict(dropout=state.rng)
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
            output = state.apply_fn(
                params_dict, x=x_n, t=t_n, ctx=context, rngs=rngs_dict, training=False)
            return output
        _logitsB = batch["images"]  # the current mode
        _logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        logitsB = normalize(_logitsB)
        begin = time.time()
        f_gen = state.sample(
            state.rng, apply, logitsB, dsb_stats)
        sec = time.time()-begin
        f_gen = unnormalize(f_gen)
        f_real = _logitsA
        f_init = _logitsB

        f_all = jnp.stack([f_init, f_gen, f_real], axis=-1)

        (
            (ens_acc, ens_nll),
            (
                (b_acc, b_nll), (a_acc, a_nll)
            )
        ) = ensemble_accuracy(
            [f_init, f_gen], labels, batch["marker"], ["B", "A"])
        count = jnp.sum(batch["marker"])
        kld = kl_divergence(f_real, f_gen, batch["marker"], "A", "A")
        rkld = kl_divergence(f_gen, f_real, batch["marker"], "A", "A")
        skld = kl_divergence(f_gen, f_init, batch["marker"], "A", "B")
        rskld = kl_divergence(f_init, f_gen, batch["marker"], "B", "A")

        metrics = OrderedDict({
            "acc": a_acc, "nll": a_nll,
            "ens_acc": ens_acc, "ens_nll": ens_nll,
            "count": count, "kld": kld,
            "rkld": rkld, "skld": skld,
            "rskld": rskld,
            "sec": sec*count
        })
        metrics = jax.lax.psum(metrics, axis_name='batch')

        return f_all, metrics

    def step_acc_ref(state, batch):
        _logitsB = batch["images"]  # the current mode
        _logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        f_real = _logitsA
        f_init = _logitsB
        (
            (ens_acc, ens_nll),
            (
                (b_acc, b_nll), (a_acc, a_nll)
            )
        ) = ensemble_accuracy(
            [f_init, f_real], labels, batch["marker"], ["B", "A"])
        rkld = kl_divergence(f_init, f_real, batch["marker"], "B", "A")
        kld = kl_divergence(f_real, f_init, batch["marker"], "A", "B")
        metrics = OrderedDict(
            {"acc_ref": a_acc, "nll_ref": a_nll,
             "ens_acc_ref": ens_acc, "ens_nll_ref": ens_nll,
             "kld_ref": kld, "rkld_ref": rkld})
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

    wandb.init(
        project="dsb-bnn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/ens_acc", summary="max")
    wandb.define_metric("val/ens_nll", summary="min")
    wandb.run.summary["params"] = sum(
        x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    wl = WandbLogger()

    for epoch_idx, _ in enumerate(tqdm(range(config.optim_ne)), start=1):
        rng = jax.random.fold_in(rng, epoch_idx)
        data_rng, rng = jax.random.split(rng)

        # -----------------------------------------------------------------
        # training
        # -----------------------------------------------------------------
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
                if (epoch_idx+1) % 50 == 0:
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
        wl.log(trn_summary)

        if state.batch_stats is not None:
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        # -----------------------------------------------------------------
        # validation
        # -----------------------------------------------------------------
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
        wl.log(val_summary)

        # -----------------------------------------------------------------
        # testing
        # -----------------------------------------------------------------
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
            wl.log(tst_summary)
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
        wl.flush()

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=5e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=5e-4, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    # ---------------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--features_dir", default="features_fixed", type=str)
    parser.add_argument("--context", action="store_true")
    parser.add_argument(
        "--contexts_dir", default="features_last_fixed", type=str)
    parser.add_argument("--gamma", default=0., type=float)
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--get_stats", action="store_true")
    parser.add_argument("--n_Amodes", default=1, type=int)
    parser.add_argument("--n_samples_each_mode", default=1, type=int)
    parser.add_argument("--take_valid", action="store_true")
    parser.add_argument("--ema_decay", default=0.999, type=float)
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=50, type=int)
    parser.add_argument("--beta1", default=5e-4, type=float)
    parser.add_argument("--beta2", default=0.02, type=float)
    # ---------------------------------------------------------------------------------------
    # networks
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--n_feat", default=256, type=int)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--droprate", default=0.2, type=float)
    # UNetModel only
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--num_res_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--use_scale_shift_norm", action="store_true")
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")

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

    settings = __import__(f"{args.features_dir}.settings", fromlist=[""])
    for k, v in vars(settings).items():
        if "__" in k:
            continue
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool):
            setattr(args, k, v)
    del settings

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
