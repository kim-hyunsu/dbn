from builtins import NotImplemented, NotImplementedError
from audioop import cross
import os
import sched
import orbax


from typing import Any, Tuple

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
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet, FlaxResNetBase
from utils import evaluate_top2acc, evaluate_topNacc, get_single_batch
from models.resnet import FlaxResNetClassifier, FlaxResNetClassifier2, FlaxResNetClassifier3, FlaxResNetClassifier4
from models.bridge import CorrectionModel, FeatureUnet, LatentFeatureUnet, dsb_schedules, MLP
from models.i2sb import ClsUnet, DiffusionBridgeNetwork, TinyUNetModel, UNetModel, MidUNetModel, DiffusionClassifier
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import WandbLogger, pixelize, normalize_logits, unnormalize_logits
from utils import model_list, logit_dir_list, feature_dir_list, feature2_dir_list, feature3_dir_list
from utils import get_info_in_dir, jprint, expand_to_broadcast, FeatureBank, get_probs, get_logprobs
from utils import batch_mul, batch_add, get_ens_logits, get_avg_logits
from tqdm import tqdm
from functools import partial
import defaults_sgd
from einops import rearrange
from flax import traverse_util


class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    ema_params: Any


def get_latentbe_state(config, rng):
    class TrainState(train_state.TrainState):
        image_stats: Any = None
        batch_stats: Any = None
    # build model

    def initialize_model(key, model, im_shape, im_dtype):
        @jax.jit
        def init(*args):
            return model.init(*args)
        var_dict = init({'params': key}, jnp.ones(im_shape, im_dtype))
        return var_dict
    model = get_resnet(config, head=True)()
    if config.data_name in ['CIFAR10_x32',]:
        image_shape = (1, 32, 32, 3,)
        num_classes = 10
    elif config.data_name in ['CIFAR100_x32',]:
        image_shape = (1, 32, 32, 3,)
        num_classes = 100
    else:
        raise NotImplementedError

    im_dtype = jnp.float32
    var_dict = initialize_model(rng, model, image_shape, im_dtype)
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=config.optim_lr,
                transition_steps=0,
            ),
            optax.cosine_decay_schedule(
                init_value=config.optim_lr,
                decay_steps=(config.optim_ne) * config.trn_steps_per_epoch,
            )
        ], boundaries=[0,]
    )
    optimizer = optax.sgd(learning_rate=scheduler, momentum=0.9, nesterov=True)
    frozen_keys = []
    partition_optimizer = {
        "trainable": optimizer,
        "frozen": optax.set_to_zero()
    }

    def include(keywords, path):
        included = False
        for k in keywords:
            if k in path:
                included = True
                break
        return included
    param_partitions = freeze(traverse_util.path_aware_map(
        lambda path, v: "frozen" if include(frozen_keys, path) else "trainable", var_dict["params"]))
    optimizer = optax.multi_transform(
        partition_optimizer, param_partitions)

    state = TrainState.create(
        apply_fn=model.apply,
        params=var_dict['params'],
        tx=optimizer,
        image_stats=var_dict["image_stats"],
        batch_stats=var_dict['batch_stats'] if 'batch_stats' in var_dict else {
        },
    )
    state = checkpoints.restore_checkpoint(
        ckpt_dir=config.latentbe,
        target=state,
        step=None,
        prefix='ckpt_e',
        parallel=True,
    )
    return state


def get_resnet(config, head=False):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet if head else FlaxResNetBase,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=config.dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=config.num_classes)

    if config.model_style == 'BN-ReLU':
        model = _ResNet
    elif config.model_style == "FRN-Swish":
        model = partial(
            _ResNet,
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)

    if head:
        return model
    return partial(model, out=config.feature_name)


def load_resnet(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["model"]["params"]
    batch_stats = ckpt["model"].get("batch_stats")
    image_stats = ckpt["model"].get("image_stats")
    return params, batch_stats, image_stats


def get_model_list(config):
    name = config.data_name
    style = config.model_style
    shared = config.shared_head
    sgd_state = getattr(config, "sgd_state", False)
    # tag_list = [
    #     "bezier", "distill", "distref", "distA",
    #     "distB", "AtoB", "AtoshB", "AtoshABC",
    #     "AtoABC"
    # ]
    tag = config.tag
    assert tag is not None, "Specify a certain group of checkpoints (--tag)"
    model_dir_list = model_list(name, style, shared, tag)
    return model_dir_list


def get_classifier(config):
    feature_name = config.feature_name
    num_classes = config.num_classes

    module = FlaxResNetClassifier3

    classifier = partial(module,
                         depth=config.model_depth,
                         widen_factor=config.model_width,
                         dtype=config.dtype,
                         pixel_mean=defaults_sgd.PIXEL_MEAN,
                         pixel_std=defaults_sgd.PIXEL_STD,
                         num_classes=num_classes,
                         feature_name=feature_name)
    if config.model_style == "BN-ReLU":
        classifier = classifier
    elif config.model_style == "FRN-Swish":
        classifier = partial(
            classifier,
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros
            ),
            norm=FilterResponseNorm,
            relu=flax.linen.swish
        )
    return classifier


def pdict(params, batch_stats=None, image_stats=None):
    params_dict = dict(params=params)
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats
    if image_stats is not None:
        params_dict["image_stats"] = image_stats
    return params_dict


def get_stats(config, dataloaders, base_net, params_dict):
    feature_name = config.feature_name

    mutable = ["intermediates"]

    @partial(jax.pmap, axis_name="batch")
    def forward_sum(batch):
        _, state = base_net.apply(
            params_dict, batch["images"], rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        feature = state["intermediates"][feature_name][0]
        feature_sum = jnp.where(
            batch["marker"][..., None, None, None],
            feature, 0
        ).sum(0)
        count = jnp.sum(batch["marker"])

        feature_sum = jax.lax.psum(feature_sum, axis_name="batch")
        count = jax.lax.psum(count, axis_name="batch")
        return feature_sum, count

    train_loader = dataloaders["trn_loader"](rng=None)
    train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
    z_dim = None
    z_sum = 0
    count = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        z, c = forward_sum(batch)
        if z_dim is None:
            z_dim = z[0].shape
        z_sum += z[0]  # first gpu
        count += c[0]  # first gpu

    z_mean = z_sum/count
    # TODO z_std
    return z_mean, z_dim


def get_scorenet(config):
    score_input_dim = config.score_input_dim
    image_size = score_input_dim[0]
    in_channels = score_input_dim[-1]
    num_channels = in_channels//config.z_dim[-1] * config.n_feat
    channel_mult = ""
    learn_sigma = config.learn_sigma
    num_res_blocks = config.num_res_blocks
    attention_resolutions = "16"
    num_heads = config.num_heads
    num_head_channels = -1
    num_heads_upsample = -1
    use_scale_shift_norm = config.use_scale_shift_norm
    dropout = config.droprate
    resblock_updown = config.resblock_updown
    use_new_attention_order = config.use_new_attention_order
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
            channel_mult = (1,)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult)
                             for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if config.joint:
        score_func = partial(
            ClsUnet,
            num_input=config.fat,
            p_dim=config.num_classes,
            z_dim=config.z_dim,
            ch=num_channels
        )
        if config.model_style == "BN-ReLU":
            score_func = score_func
        elif config.model_style == "FRN-Swish":
            score_func = partial(
                score_func,
                conv=partial(
                    flax.linen.Conv,
                    use_bias=True,
                    kernel_init=jax.nn.initializers.he_normal(),
                    bias_init=jax.nn.initializers.zeros
                ),
                norm=FilterResponseNorm,
                relu=flax.linen.swish
            )
        else:
            raise NotImplementedError
    else:
        score_func = partial(
            TinyUNetModel,
            ver=config.version,
            image_size=image_size,  # 8
            in_channels=in_channels,  # 128
            model_channels=num_channels,  # 256
            out_channels=(
                in_channels if not learn_sigma else 2 * in_channels),  # 128
            num_res_blocks=num_res_blocks,  # 1
            attention_resolutions=tuple(attention_ds),  # (0,)
            dropout=dropout,
            channel_mult=channel_mult,  # (1,)
            num_classes=None,
            dtype=config.dtype,
            num_heads=num_heads,  # 1
            num_head_channels=num_head_channels,  # -1
            num_heads_upsample=num_heads_upsample,  # -1
            use_scale_shift_norm=use_scale_shift_norm,  # False
            resblock_updown=resblock_updown,  # False
            use_new_attention_order=use_new_attention_order
        )
    return score_func


def get_score_input_dim(config):
    if config.fat > 1:
        if isinstance(config.z_dim, tuple):
            h, w, d = config.z_dim
            score_input_dim = (h, w, config.fat*d)
        else:
            score_input_dim = config.fat*config.z_dim
    else:
        score_input_dim = config.z_dim

    return score_input_dim


def load_base_cls(variables, resnet_param_list, load_cls=True, base_type="A"):
    def sorter(x):
        assert "_" in x
        name, num = x.split("_")
        return (name, int(num))

    if base_type == "A":
        def get(key1, key2):
            return resnet_param_list[0][key1][key2]
    elif base_type == "AVG":
        def get(key1, key2):
            return jax.tree_util.tree_map(
                lambda *args: sum(args)/len(args),
                *[param[key1][key2] for param in resnet_param_list]
            )
    elif base_type == "BE":
        be_params = resnet_param_list.pop(-1)

        def get(key1, key2):
            return be_params[key1][key2]
    else:
        raise NotImplementedError

    resnet_params = resnet_param_list[0]
    var = variables.unfreeze()
    base_param_keys = []
    res_param_keys = []
    cls_param_keys = []
    for k, v in resnet_params["params"].items():
        res_param_keys.append(k)
    for k, v in var["params"]["base"].items():
        base_param_keys.append(k)
    for k, v in var["params"]["cls"].items():
        cls_param_keys.append(k)
    res_param_keys = sorted(res_param_keys, key=sorter)
    base_param_keys = sorted(base_param_keys, key=sorter)
    cls_param_keys = sorted(cls_param_keys, key=sorter)

    cls_idx = 0
    for k in res_param_keys:
        if k in base_param_keys:
            var["params"]["base"][k] = get("params", k)
        else:
            cls_k = cls_param_keys[cls_idx]
            if load_cls:
                var["params"]["cls"][cls_k] = get("params", k)
                for key, value in var["params"]["cls"][cls_k].items():
                    assert jnp.all(value == resnet_params["params"][k][key])
            cls_idx += 1

    isbatchnorm = resnet_params.get("batch_stats") is not None
    if isbatchnorm:
        base_batch_keys = []
        res_batch_keys = []
        cls_batch_keys = []
        for k, v in resnet_params["batch_stats"].items():
            res_batch_keys.append(k)
        for k, v in var["batch_stats"]["base"].items():
            base_batch_keys.append(k)
        for k, v in var["batch_stats"]["cls"].items():
            cls_batch_keys.append(k)
        res_batch_keys = sorted(res_batch_keys, key=sorter)
        base_batch_keys = sorted(base_batch_keys, key=sorter)
        cls_batch_keys = sorted(cls_batch_keys, key=sorter)

        cls_idx = 0
        for k in res_batch_keys:
            if k in base_batch_keys:
                var["batch_stats"]["base"][k] = get("batch_stats", k)
            else:
                cls_k = cls_batch_keys[cls_idx]
                if load_cls:
                    var["batch_stats"]["cls"][cls_k] = get("batch_stats", k)
                cls_idx += 1

    return freeze(var)


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    model_dtype = jnp.float32
    config.dtype = model_dtype
    config.image_stats = dict(
        m=jnp.array(defaults_sgd.PIXEL_MEAN),
        s=jnp.array(defaults_sgd.PIXEL_STD))

    # ------------------------------------------------------------------------
    # load image dataset (C10, C100, TinyImageNet, ImageNet)
    # ------------------------------------------------------------------------
    dataloaders = build_dataloaders(config)
    config.num_classes = dataloaders["num_classes"]
    config.trn_steps_per_epoch = dataloaders["trn_steps_per_epoch"]

    # ------------------------------------------------------------------------
    # define and load resnet
    # ------------------------------------------------------------------------
    model_dir_list = get_model_list(config)
    resnet_param_list = []
    assert len(
        model_dir_list) >= config.fat, "# of checkpoints is insufficient than config.fat"
    for dir in model_dir_list[:config.fat]:
        params, batch_stats, image_stats = load_resnet(dir)
        resnet_params = pdict(
            params=params, batch_stats=batch_stats, image_stats=image_stats)
        resnet_param_list.append(resnet_params)
    if config.base_type.upper() == "BE":
        assert config.latentbe
        _state = get_latentbe_state(config, rng)
        resnet_params = pdict(
            params=_state.params,
            batch_stats=_state.batch_stats,
            image_stats=_state.image_stats
        )
        resnet_param_list.append(resnet_params)

        @partial(jax.pmap, axis_name="batch")
        def test_model(batch):
            _, new_model_state = _state.apply_fn(
                resnet_params, batch['images'],
                rngs=None,
                mutable='intermediates',
                use_running_average=True)
            logits = new_model_state['intermediates']['cls.logit'][0]
            predictions = jax.nn.log_softmax(
                logits, axis=-1)  # [B, K,]
            target = common_utils.onehot(
                batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
            loss = -jnp.sum(target * predictions, axis=-1)      # [B,]
            acc = evaluate_acc(
                predictions, batch['labels'], log_input=True, reduction='none')
            nll = evaluate_nll(
                predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

            # refine and return metrics
            loss = jnp.sum(
                jnp.where(batch['marker'], loss, jnp.zeros_like(loss)))
            acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
            nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
            cnt = jnp.sum(batch['marker'])
            metrics = OrderedDict(
                {"loss": loss, 'acc': acc, 'nll': nll, 'cnt': cnt})
            metrics = jax.lax.psum(metrics, axis_name='batch')
            return metrics
        tst_metric = []
        tst_loader = dataloaders['tst_loader'](rng=None)
        tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
        for batch_idx, batch in enumerate(tst_loader, start=1):
            metrics = test_model(batch)
            tst_metric.append(metrics)
        tst_metric = common_utils.get_metrics(tst_metric)
        tst_summarized = {
            f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
        tst_summarized['tst/loss'] /= tst_summarized['tst/cnt']
        tst_summarized['tst/acc'] /= tst_summarized['tst/cnt']
        tst_summarized['tst/nll'] /= tst_summarized['tst/cnt']
        print("Latent BE")
        for k, v in tst_summarized.items():
            print(k, v)
    # ------------------------------------------------------------------------
    # determine dims of base/score/cls
    # ------------------------------------------------------------------------
    resnet = get_resnet(config, head=True)
    resnet = resnet()
    cls_net = get_classifier(config)
    _, h, w, d = dataloaders["image_shape"]
    x_dim = (h, w, d)
    config.x_dim = x_dim
    print("Calculating statistics of feature z...")
    z_mean, z_dim = get_stats(
        config, dataloaders, resnet, resnet_param_list[0])
    print(f"z mean: {z_mean.mean():.3f}")
    config.z_dim = z_dim
    score_input_dim = get_score_input_dim(config)
    config.score_input_dim = score_input_dim

    @jax.jit
    def forward_resnet(params_dict, batch):
        mutable = ["intermediates"]
        _, state = resnet.apply(
            params_dict, batch["images"], rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        features = state["intermediates"][config.feature_name][0]
        logits = state["intermediates"]["cls.logit"][0]
        return features, logits

    # ------------------------------------------------------------------------
    # define score and cls
    # ------------------------------------------------------------------------
    print("Building Diffusion Bridge Network (DBN)...")
    base_net = get_resnet(config, head=False)
    dsb_stats = dsb_schedules(config.beta1, config.beta2, config.T)
    dsb_stats["n_T"] = config.T
    score_net = get_scorenet(config)
    crt_net = partial(CorrectionModel, layers=1) if config.crt > 0 else None
    dbn = DiffusionBridgeNetwork(
        base_net=base_net,
        score_net=score_net,
        cls_net=cls_net,
        crt_net=crt_net,
        dsb_stats=dsb_stats,
        fat=config.fat,
        joint=config.joint
    )

    # ------------------------------------------------------------------------
    # initialize score & cls and replace base and cls with loaded params
    # ------------------------------------------------------------------------
    print("Initializing DBN...")
    init_rng, sub_rng = jax.random.split(rng)
    if config.joint:
        variables = dbn.init(
            {"params": init_rng},
            rng=init_rng,
            lz0=(
                jnp.empty((1, config.fat*config.num_classes)),
                jnp.empty((1, *score_input_dim))
            ),
            x1=jnp.empty((1, *x_dim)),
            training=False
        )
    else:
        variables = dbn.init(
            {"params": init_rng},
            rng=init_rng,
            z0=jnp.empty((1, *score_input_dim)),
            x1=jnp.empty((1, *x_dim)),
            training=False
        )
    print("Loading base and cls networks...")
    variables = load_base_cls(
        variables, resnet_param_list,
        load_cls=not config.cls_from_scratch,
        base_type=config.base_type.upper()
    )
    config.image_stats = variables["image_stats"]

    # ------------------------------------------------------------------------
    # define optimizers
    # ------------------------------------------------------------------------
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.warmup_factor*config.optim_lr,
        peak_value=config.optim_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.optim_ne * config.trn_steps_per_epoch)
    cls_scheduler = [
        optax.constant_schedule(0), optax.constant_schedule(1)]
    cls_boundaries = [config.start_cls*config.trn_steps_per_epoch]
    cls_fn = optax.join_schedules(
        schedules=cls_scheduler,
        boundaries=cls_boundaries
    )
    if config.optim_base == "adam":
        optim = partial(
            optax.adamw, learning_rate=scheduler, weight_decay=config.optim_weight_decay)
    elif config.optim_base == "sgd":
        optim = partial(
            optax.sgd, learning_rate=scheduler, momentum=config.optim_momentum)
    else:
        raise NotImplementedError

    partition_optimizers = {
        "base": optax.set_to_zero() if not config.train_base else optim(),
        "score": optim(),
        "cls": optim(learning_rate=lambda step: cls_fn(step)*scheduler(step)),
        "crt": optim()
    }

    def tagging(path, v):
        if "base" in path:
            return "base"
        elif "score" in path:
            return "score"
        elif "cls" in path:
            return "cls"
        elif "crt" in path:
            return "crt"
        else:
            raise NotImplementedError
    partitions = flax.core.freeze(
        flax.traverse_util.path_aware_map(tagging, variables["params"]))
    optimizer = optax.multi_transform(partition_optimizers, partitions)

    # ------------------------------------------------------------------------
    # create train state
    # ------------------------------------------------------------------------
    state_rng, sub_rng = jax.random.split(sub_rng)
    state = TrainState.create(
        apply_fn=dbn.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=state_rng
    )

    # ------------------------------------------------------------------------
    # define sampler and metrics
    # ------------------------------------------------------------------------
    def dsb_sample(score, rng, x0, y0=None):
        shape = x0.shape
        batch_size = shape[0]
        _sigma_t = dsb_stats["sigma_t"]
        _alpos_weight_t = dsb_stats["alpos_weight_t"]
        _sigma_t_square = dsb_stats["sigma_t_square"]
        n_T = dsb_stats["n_T"]
        _t = jnp.array([1/n_T])
        _t = jnp.tile(_t, [batch_size])
        std_arr = jnp.sqrt(_alpos_weight_t*_sigma_t_square)
        h_arr = jax.random.normal(rng, (len(_sigma_t), *shape))
        h_arr = h_arr.at[0].set(0)
        if y0 is not None:
            y_shape = y0.shape
            y_h_arr = jax.random.normal(rng, (len(_sigma_t), *y_shape))
            y_h_arr = y_h_arr.at[0].set(0)

        @jax.jit
        def body_fn(n, val):
            x_n = val
            idx = n_T - n
            t_n = idx * _t

            h = h_arr[idx]  # (B, d)
            eps = score(x_n, t=t_n)  # (2*B, d)

            sigma_t = _sigma_t[idx]
            alpos_weight_t = _alpos_weight_t[idx]
            std = std_arr[idx]

            x_0_eps = x_n - sigma_t*eps
            mean = alpos_weight_t*x_0_eps + (1-alpos_weight_t)*x_n
            x_n = mean + std * h  # (B, d)

            return x_n

        @jax.jit
        def joint_fn(n, val):
            p_n, x_n = val
            idx = n_T - n
            t_n = idx * _t

            p_h = h_arr[idx]  # (B, d)
            x_h = y_h_arr[idx]  # (B, d)
            peps, xeps = score(p_n, x_n, t=t_n)  # (2*B, d)

            sigma_t = _sigma_t[idx]
            alpos_weight_t = _alpos_weight_t[idx]
            std = std_arr[idx]

            p_0_eps = p_n - sigma_t*peps
            x_0_eps = x_n - sigma_t*xeps
            p_mean = alpos_weight_t*p_0_eps + (1-alpos_weight_t)*p_n
            x_mean = alpos_weight_t*x_0_eps + (1-alpos_weight_t)*x_n
            x_n = x_mean + std * x_h  # (B, d)
            p_n = p_mean + std * p_h  # (B, d)

            return p_n, x_n

        if y0 is not None:
            val = (x0, y0)
            for i in range(0, n_T):
                val = joint_fn(i, val)
            x_n, y_0 = val
            return x_n, y_0

        val = x0
        for i in range(0, n_T):
            val = body_fn(i, val)
        x_n = val

        return x_n

    @jax.jit
    def mse_loss(noise, output):
        p = config.mse_power
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.abs(noise-output)**p, axis=sum_axis)
        return loss

    def kld_loss(target_list, refer_list):
        if not isinstance(target_list, list):
            target_list = [target_list]
        if not isinstance(refer_list, list):
            refer_list = [refer_list]
        kld_sum = 0
        count = 0
        for tar, ref in zip(target_list, refer_list):
            assert len(tar.shape) == 2, f"{tar.shape}"
            assert len(ref.shape) == 2, f"{ref.shape}"
            logq = jax.nn.log_softmax(tar, axis=-1)
            logp = jax.nn.log_softmax(ref, axis=-1)
            q = jnp.exp(logq)
            integrand = q*(logq-logp)
            kld = jnp.sum(integrand, axis=-1)
            kld_sum += kld
            count += 1
        return kld_sum/count

    @jax.jit
    def reduce_mean(loss, marker):
        assert len(loss.shape) == 1
        count = jnp.sum(marker)
        loss = jnp.where(marker, loss, 0).sum()
        loss = jnp.where(count != 0, loss/count, loss)
        return loss

    @jax.jit
    def reduce_sum(loss, marker):
        assert len(loss.shape) == 1
        loss = jnp.where(marker, loss, 0).sum()
        return loss

    def kld_loss_fn(t, r):
        return reduce_sum(
            kld_loss(t, r), batch["marker"])

    # ------------------------------------------------------------------------
    # define step collecting features and logits
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_label(batch):
        z0_list = []
        logitsA = []
        for res_params_dict in resnet_param_list:
            z0, logits0 = forward_resnet(res_params_dict, batch)
            z0_list.append(z0)
            logitsA.append(logits0)
        zB = z0_list[0]
        zA = jnp.concatenate(z0_list, axis=-1)
        logitsB = logitsA[0]
        batch["zB"] = zB
        batch["zA"] = zA
        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        return batch

    # ------------------------------------------------------------------------
    # define train step
    # ------------------------------------------------------------------------
    # jit is not allowed
    def loss_func(params, state, batch, train=True):
        zA = batch["zA"]
        logitsA = batch["logitsA"]
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        output = state.apply_fn(
            params_dict, score_rng,
            zA, batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        (epsilon, z_t, t, mu_t,
         sigma_t), logits0eps = output[0] if train else output
        _sigma_t = expand_to_broadcast(sigma_t, z_t, axis=1)
        diff = (z_t-zA) / _sigma_t
        score_loss = mse_loss(epsilon, diff)
        logits0eps = [l for l in logits0eps]
        cls_loss = kld_loss(logits0eps, logitsA) / sigma_t**2

        count = jnp.sum(batch["marker"])
        score_loss = reduce_mean(score_loss, batch["marker"])
        cls_loss = reduce_mean(cls_loss, batch["marker"])

        total_loss = config.gamma*score_loss + config.beta*cls_loss

        metrics = OrderedDict({
            "loss": total_loss*count,
            "score_loss": score_loss*count,
            "cls_loss": cls_loss*count,
            "count": count
        })

        return total_loss, (metrics, new_model_state)

    # jit is not allowed
    def loss_func_joint(params, state, batch, train=True):
        zA = batch["zA"]
        logitsA = batch["logitsA"]  # length fat list
        _logitsA = jnp.concatenate(logitsA, axis=-1)  # (B, fat*num_classes)
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        output = state.apply_fn(
            params_dict, score_rng,
            (_logitsA, zA), batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        (
            (leps, zeps), (l_t, z_t), t, mu_t, sigma_t
        ), l0eps = output[0] if train else output

        z_sigma_t = expand_to_broadcast(sigma_t, z_t, axis=1)
        z_diff = (z_t-zA) / z_sigma_t
        l_sigma_t = expand_to_broadcast(sigma_t, l_t, axis=1)
        l_diff = (l_t-_logitsA) / l_sigma_t
        z_score_loss = mse_loss(zeps, z_diff)
        l_score_loss = mse_loss(leps, l_diff)
        score_loss = z_score_loss + l_score_loss

        count = jnp.sum(batch["marker"])
        score_loss = reduce_mean(score_loss, batch["marker"])
        if config.kld_joint:
            l0eps = [l for l in l0eps]
            cls_loss = kld_loss(l0eps, logitsA) / sigma_t**2
            cls_loss = reduce_mean(cls_loss, batch["marker"])
        else:
            cls_loss = 0

        total_loss = config.gamma*score_loss + config.beta*cls_loss

        metrics = OrderedDict({
            "loss": total_loss*count,
            "score_loss": score_loss*count,
            "cls_loss": cls_loss*count,
            "count": count
        })

        return total_loss, (metrics, new_model_state)

    @partial(jax.pmap, axis_name="batch")
    def step_train(state, batch):

        def loss_fn(params):
            _loss_func = loss_func_joint if config.joint else loss_func
            return _loss_func(params, state, batch)

        (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state.get("batch_stats"))
        a = config.ema_decay
        def update_ema(wt, ema_tm1): return jnp.where(
            (wt != ema_tm1) & (a < 1), a*wt + (1-a)*ema_tm1, wt)
        new_state = new_state.replace(
            ema_params=jax.tree_util.tree_map(
                update_ema,
                new_state.params,
                new_state.ema_params))
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return new_state, metrics

    # ------------------------------------------------------------------------
    # define valid step
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_valid(state, batch):
        _loss_func = loss_func_joint if config.joint else loss_func
        _, (metrics, _) = _loss_func(
            state.ema_params, state, batch, train=False)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    # ------------------------------------------------------------------------
    # define sampling step
    # ------------------------------------------------------------------------
    def ensemble_accnll(logits, labels, marker):
        acc_list = []
        nll_list = []
        cum_acc_list = []
        cum_nll_list = []
        prob_sum = 0
        for i, lg in enumerate(logits):
            logprob = jax.nn.log_softmax(lg, axis=-1)
            prob = jnp.exp(logprob)
            prob_sum += prob
            acc = evaluate_acc(
                logprob, labels, log_input=True, reduction="none")
            nll = evaluate_nll(
                logprob, labels, log_input=True, reduction='none')
            acc = reduce_sum(acc, marker)
            nll = reduce_sum(nll, marker)
            acc_list.append(acc)
            nll_list.append(nll)
            if i != 0:  # Not B
                avg_prob = prob_sum / (i+1)
                acc = evaluate_acc(
                    avg_prob, labels, log_input=False, reduction="none")
                nll = evaluate_nll(
                    avg_prob, labels, log_input=False, reduction='none')
                acc = reduce_sum(acc, marker)
                nll = reduce_sum(nll, marker)
                cum_acc_list.append(acc)
                cum_nll_list.append(nll)
        return acc_list, nll_list, cum_acc_list, cum_nll_list

    def sample_func(state, batch):
        logitsB = batch["logitsB"]
        logitsA = batch["logitsA"]
        labels = batch["labels"]
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        logitsC = model_bd.sample(score_rng, dsb_sample, batch["images"])
        logitsC = [l for l in logitsC]

        kld = kld_loss_fn(logitsA, logitsC)
        rkld = kld_loss_fn(logitsC, logitsA)
        skld = kld_loss_fn(logitsC, [logitsB]*len(logitsC))
        rskld = kld_loss_fn([logitsB]*len(logitsC), logitsC)
        (
            acc_list, nll_list, cum_acc_list, cum_nll_list
        ) = ensemble_accnll(logitsC, labels, batch["marker"])
        metrics = OrderedDict({
            "kld": kld, "rkld": rkld,
            "skld": skld, "rskld": rskld,
            "count": jnp.sum(batch["marker"]),
            "acc0": acc_list[0], "nll0": nll_list[0]
        })

        for i, (acc, nll, ensacc, ensnll) in enumerate(
                zip(acc_list[1:], nll_list[1:], cum_acc_list, cum_nll_list), start=1):
            metrics[f"acc{i}"] = acc
            metrics[f"nll{i}"] = nll
            metrics[f"ens_acc{i}"] = ensacc
            metrics[f"ens_nll{i}"] = ensnll

        return metrics

    @partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        metrics = sample_func(state, batch)

        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    @jax.jit
    def accnll(logits, labels, marker):
        logprob = jax.nn.log_softmax(logits, axis=-1)
        acc = evaluate_acc(
            logprob, labels, log_input=True, reduction="none")
        nll = evaluate_nll(
            logprob, labels, log_input=True, reduction='none')
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        return acc, nll

    @partial(jax.pmap, axis_name="batch")
    def step_acc_ref(state, batch):
        logitsB = batch["logitsB"]
        logitsA = batch["logitsA"]
        labels = batch["labels"]

        kld = kld_loss_fn([logitsB]*len(logitsA), logitsA)
        rkld = kld_loss_fn(logitsA, [logitsB]*len(logitsA))
        acc_from, nll_from = accnll(logitsB, labels, batch["marker"])
        (
            acc_list, nll_list, cum_acc_list, cum_nll_list
        ) = ensemble_accnll(logitsA, labels, batch["marker"])
        metrics = OrderedDict({
            "kld_ref": kld, "rkld_ref": rkld,
            "acc_from": acc_from, "nll_from": nll_from,
            "count": jnp.sum(batch["marker"]),
            "acc0": acc_list[0], "nll0": nll_list[0]
        })
        for i, (acc, nll, ensacc, ensnll) in enumerate(
                zip(acc_list[1:], nll_list[1:], cum_acc_list, cum_nll_list), start=1):
            metrics[f"acc{i}_ref"] = acc
            metrics[f"nll{i}_ref"] = nll
            metrics[f"ens_acc{i}_ref"] = ensacc
            metrics[f"ens_nll{i}_ref"] = ensnll

        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    # ------------------------------------------------------------------------
    # define mixup
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_mixup(state, batch):
        count = jnp.sum(batch["marker"])
        x = batch["images"]
        y = batch["labels"]
        batch_size = x.shape[0]
        a = config.mixup_alpha
        beta_rng, perm_rng = jax.random.split(state.rng)
        lamda = jnp.where(a > 0, jax.random.beta(beta_rng, a, a), 1)
        perm_x = jax.random.permutation(perm_rng, x)
        perm_y = jax.random.permutation(perm_rng, y)
        mixed_x = lamda*x+(1-lamda)*perm_x
        mixed_y = jnp.where(lamda >= 0.5, y, perm_y)
        mixed_x = jnp.where(count == batch_size, mixed_x, x)
        mixed_y = jnp.where(count == batch_size, mixed_y, y)
        batch["images"] = mixed_x
        batch["labels"] = mixed_y
        return batch

    # ------------------------------------------------------------------------
    # init settings and wandb
    # ------------------------------------------------------------------------
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    best_acc = -float("inf")
    train_summary = dict()
    valid_summary = dict()
    test_summary = dict()
    state = jax_utils.replicate(state)

    wandb.init(
        project="dbn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/ens_acc", summary="max")
    wandb.define_metric("val/ens_nll", summary="min")
    wandb.define_metric(f"val/ens_acc{config.fat-1}", summary="max")
    wandb.define_metric(f"val/ens_nll{config.fat-1}", summary="min")
    wandb.run.summary["params_base"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["base"]))
    )
    wandb.run.summary["params_score"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["score"]))
    )
    wandb.run.summary["params_cls"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["cls"]))
    )
    if config.crt > 0:
        wandb.run.summary["params_crt"] = (
            sum(x.size for x in jax.tree_util.tree_leaves(
                variables["params"]["crt"]))
        )
    wl = WandbLogger()

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), metrics).items()}
        for k, v in summarized.items():
            if "count" in k:
                continue
            elif "lr" in k:
                continue
            summarized[k] /= summarized[f"{key}/count"]
        del summarized[f"{key}/count"]
        return summarized

    for epoch_idx in tqdm(range(config.optim_ne)):
        epoch_rng = jax.random.fold_in(sub_rng, epoch_idx)
        train_rng, valid_rng, test_rng = jax.random.split(epoch_rng, 3)

        # ------------------------------------------------------------------------
        # train by getting features from each resnet
        # ------------------------------------------------------------------------
        train_loader = dataloaders["dataloader"](rng=epoch_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        train_metrics = []
        for batch_idx, batch in enumerate(train_loader):
            batch_rng = jax.random.fold_in(train_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            if config.mixup_alpha > 0:
                batch = step_mixup(state, batch)
            batch = step_label(batch)
            state, metrics = step_train(state, batch)
            if epoch_idx == 0:
                acc_ref_metrics = step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if epoch_idx % 10 == 0:
                acc_metrics = step_sample(state, batch)
                metrics.update(acc_metrics)
            train_metrics.append(metrics)

        train_summarized = summarize_metrics(train_metrics, "trn")
        train_summary.update(train_summarized)
        wl.log(train_summary)

        if state.batch_stats is not None:
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        # ------------------------------------------------------------------------
        # valid by getting features from each resnet
        # ------------------------------------------------------------------------
        valid_loader = dataloaders["val_loader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        valid_metrics = []
        for batch_idx, batch in enumerate(valid_loader):
            batch_rng = jax.random.fold_in(valid_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            batch = step_label(batch)
            metrics = step_valid(state, batch)
            if epoch_idx == 0:
                acc_ref_metrics = step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            acc_metrics = step_sample(state, batch)
            metrics.update(acc_metrics)
            valid_metrics.append(metrics)

        valid_summarized = summarize_metrics(valid_metrics, "val")
        valid_summary.update(valid_summarized)
        wl.log(valid_summary)

        # ------------------------------------------------------------------------
        # test by getting features from each resnet
        # ------------------------------------------------------------------------
        if config.best_nll:
            criteria = -valid_summarized[f"val/ens_nll{config.fat-1}"]
        else:
            criteria = valid_summarized[f"val/ens_acc{config.fat-1}"]
        if best_acc < criteria:
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(test_loader):
                batch_rng = jax.random.fold_in(test_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                batch = step_label(batch)
                metrics = step_valid(state, batch)
                if best_acc == -float('inf'):
                    acc_ref_metrics = step_acc_ref(state, batch)
                    metrics.update(acc_ref_metrics)
                acc_metrics = step_sample(state, batch)
                metrics.update(acc_metrics)
                test_metrics.append(metrics)

            test_summarized = summarize_metrics(test_metrics, "tst")
            test_summary.update(test_summarized)
            wl.log(test_summary)
            best_acc = criteria

        wl.flush()

        jax.random.normal(rng, ()).block_until_ready()
        if jnp.isnan(train_summarized["trn/loss"]):
            print("NaN detected")
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            arg_defaults = yaml.safe_load(f)
    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=2e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--warmup_factor', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=10, type=int)
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.1, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument('--optim_base', default="adam", type=str)
    parser.add_argument('--start_cls', default=-1, type=int)
    parser.add_argument('--cls_optim', default="adam", type=str)
    parser.add_argument('--start_base', default=999999999999, type=int)
    # ---------------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument("--tag", default=None, type=str)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--ema_decay", default=0.9999, type=float)
    parser.add_argument("--mse_power", default=2, type=int)
    parser.add_argument("--crt", default=0, type=int)
    parser.add_argument("--cls_from_scratch", action="store_true")
    parser.add_argument("--mixup_alpha", default=0, type=float)
    # ---------------------------------------------------------------------------------------
    # experiemnts
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--train_base", action="store_true")
    parser.add_argument("--base_type", default="A", type=str)
    parser.add_argument("--best_nll", action="store_true")
    # replace base with LatentBE-trained model (directory of checkpoint)
    parser.add_argument("--latentbe", default=None, type=str)
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=3e-4, type=float)
    # Fat DSB (A+A -> B1+B2)
    parser.add_argument("--fat", default=1, type=int)
    # joint diffusion
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--kld_joint", action="store_true")
    # Stochastic Optimal Control
    parser.add_argument("--soc", action="store_true")
    # ---------------------------------------------------------------------------------------
    # networks
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--n_feat", default=256, type=int)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--droprate", default=0.2, type=float)
    parser.add_argument("--feature_name", default="feature.layer3stride2")
    # UNetModel only
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--num_res_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--use_scale_shift_norm", action="store_true")
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")
    parser.add_argument("--large", action="store_true")

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    # if args.save is not None:
    #     args.save = os.path.abspath(args.save)
    #     if os.path.exists(args.save):
    #         raise AssertionError(f'already existing args.save = {args.save}')
    #     os.makedirs(args.save, exist_ok=True)

    print_fn = partial(print, flush=True)
    # if args.save:
    #     def print_fn(s):
    #         with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
    #             fp.write(s + '\n')
    #         print(s, flush=True)

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
    # import traceback
    # with open("error.log", "w") as log:
    #     try:
    #         main()
    #     except Exception:
    #         print("ERROR OCCURED! Check it out in error.log.")
    #         traceback.print_exc(file=log)
    main()
