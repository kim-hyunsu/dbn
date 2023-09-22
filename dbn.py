# generic
from builtins import NotImplementedError
import os
import orbax
from copy import deepcopy
import datetime
import wandb
from tabulate import tabulate
import sys
from collections import OrderedDict
from tqdm import tqdm
from functools import partial
from einops import rearrange
from typing import Any

# jax-related
import jax, flax, optax, jaxlib
from flax.training import common_utils, checkpoints, train_state
from flax.core.frozen_dict import freeze
from flax import jax_utils
import jax.numpy as jnp
import numpy as np

# user-defined
from default_args import defaults_sgd, defaults_dsb
from data.build import build_dataloaders
from giung2.metrics import evaluate_acc, evaluate_nll
from models.bridge import CorrectionModel, dsb_schedules
from models.i2sb import DiffusionBridgeNetwork
from utils import WandbLogger, pdict, print_fn, load_ckpt, expand_to_broadcast, batch_mul, get_stats, get_score_input_dim
from models import utils as mutils


def build_dbn(config):
    cls_net = mutils.get_classifier(config)
    base_net = mutils.get_resnet(config, head=False)
    dsb_stats = dsb_schedules(
        config.beta1, config.beta2, config.T, linear_noise=config.linear_noise, continuous=config.dsb_continuous)
    score_net = mutils.get_scorenet(config)
    crt_net = partial(CorrectionModel, layers=1) if config.crt > 0 else None
    dbn = DiffusionBridgeNetwork(
        base_net=base_net,
        score_net=score_net,
        cls_net=cls_net,
        crt_net=crt_net,
        dsb_stats=dsb_stats,
        z_dsb_stats=None,
        fat=config.fat,
        joint=2,
        forget=config.forget,
        temp=1.,
        print_inter=False,
        mimo_cond=config.mimo_cond,
        start_temp=config.start_temp,
        multi_mixup=False,
        continuous=config.dsb_continuous,
        rand_temp=config.distribution == 3
    )
    return dbn, (dsb_stats, None)


def dsb_sample(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
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

    @jax.jit
    def body_fn(n, val):
        x_n = val
        idx = n_T - n
        t_n = idx * _t

        h = h_arr[idx]  # (B, d)
        if config.mimo_cond:
            t_n = jnp.tile(t_n[:, None], reps=[1, config.fat])
        eps = score(x_n, y0, t=t_n)

        sigma_t = _sigma_t[idx]
        alpos_weight_t = _alpos_weight_t[idx]
        std = std_arr[idx]

        x_0_eps = x_n - sigma_t*eps
        mean = alpos_weight_t*x_0_eps + (1-alpos_weight_t)*x_n
        x_n = mean + std * h  # (B, d)

        return x_n

    x_list = [x0]
    val = x0
    if steps is None:
        steps = n_T
    for i in range(0, steps):
        val = body_fn(i, val)
        x_list.append(val)
    x_n = val

    return jnp.concatenate(x_list, axis=0)


def dsb_sample_cont(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
    shape = x0.shape
    batch_size = shape[0]

    timesteps = jnp.concatenate(
        [jnp.linspace(1.0, 0.001, steps), jnp.zeros([1])])

    @jax.jit
    def body_fn(n, val):
        rng, x_n, x_list = val
        t, next_t = timesteps[n], timesteps[n+1]
        vec_t = jnp.ones([batch_size]) * t
        vec_next_t = jnp.ones([batch_size]) * next_t

        if config.mimo_cond:
            vec_t = jnp.tile(vec_t[:, None], reps=[1, config.fat])
        eps = score(x_n, y0, t=vec_t)

        coeffs = dsb_stats((vec_next_t, vec_t), mode='sampling')
        x_0_eps = x_n - batch_mul(coeffs['sigma_t'], eps)
        mean = batch_mul(coeffs['x0'], x_0_eps) + batch_mul(coeffs['x1'], x_n)

        rng, step_rng = jax.random.split(rng)
        h = jax.random.normal(step_rng, x_n.shape)
        x_n = mean + batch_mul(coeffs['noise'], h)

        x_list.pop(0)
        x_list.append(x_n)
        return rng, x_n, x_list

    x_list = [x0] * (steps + 1)
    # _, _, x_list = jax.lax.fori_loop(0, steps, body_fn, (rng, x0, x_list))
    val = (rng, x0, x_list)
    for i in range(0, steps):
        val = body_fn(i, val)
    _, _, x_list = val
    return jnp.concatenate(x_list, axis=0)


def launch_train(config):
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
    model_dir_list = mutils.get_model_list(config)
    resnet_param_list = []
    assert len(
        model_dir_list) >= config.fat, "# of checkpoints is insufficient than config.fat"
    if config.ensemble_prediction == 0:
        target_model_list = model_dir_list[:config.fat+1]
    else:
        assert config.fat == 1, f"config.fat should be 1 but {config.fat} is given"
        target_model_list = model_dir_list[:config.ensemble_prediction]

    for dir in target_model_list:
        params, batch_stats, image_stats = mutils.load_resnet(dir)
        resnet_params = pdict(
            params=params, batch_stats=batch_stats, image_stats=image_stats)
        resnet_param_list.append(resnet_params)
    # ------------------------------------------------------------------------
    # determine dims of base/score/cls
    # ------------------------------------------------------------------------
    resnet = mutils.get_resnet(config, head=True)()
    _, h, w, d = dataloaders["image_shape"]
    x_dim = (h, w, d)
    config.x_dim = x_dim
    print_fn("Calculating statistics of feature z...")
    (
        (lmean, lstd, lmax, lmin, ldim),
        (zmean, zstd, zmax, zmin, zdim)
    ) = get_stats(
        config, dataloaders, resnet, resnet_param_list[0])
    print_fn(f"l mean {lmean:.3f} std {lstd:.3f} max {lmax:.3f} min {lmin:.3f} dim {ldim}")
    print_fn(f"z mean {zmean:.3f} std {zstd:.3f} max {zmax:.3f} min {zmin:.3f} dim {zdim}")
    config.z_dim = zdim
    score_input_dim = get_score_input_dim(config)
    config.score_input_dim = score_input_dim

    @jax.jit
    def forward_resnet(params_dict, images):
        mutable = ["intermediates"]
        _, state = resnet.apply(
            params_dict, images, rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        features = state["intermediates"][config.feature_name][0]
        logits = state["intermediates"]["cls.logit"][0]
        return features, logits

    # ------------------------------------------------------------------------
    # define score and cls
    # ------------------------------------------------------------------------
    print_fn("Building Diffusion Bridge Network (DBN)...")
    dbn, (dsb_stats, z_dsb_stats) = build_dbn(config)

    # ------------------------------------------------------------------------
    # initialize score & cls and replace base and cls with loaded params
    # ------------------------------------------------------------------------
    print_fn("Initializing DBN...")
    init_rng, sub_rng = jax.random.split(rng)
    variables = dbn.init(
        {"params": init_rng, "dropout": init_rng},
        rng=init_rng,
        l0=jnp.empty((1, config.fat*config.num_classes)),
        x1=jnp.empty((1, *x_dim)),
        training=False
    )
    print_fn("Loading base and cls networks...")
    variables = mutils.load_base_cls(
        variables, resnet_param_list,
        load_cls=not config.cls_from_scratch,
        base_type=config.base_type.upper(),
        mimo=1
    )
    config.image_stats = variables["image_stats"]
    if config.distribution == 2:
        _variables = deepcopy(variables)
        base_params_list, cls_params_list = mutils.get_base_cls(
            _variables, resnet_param_list)
        del _variables

    # ------------------------------------------------------------------------
    # define optimizers
    # ------------------------------------------------------------------------
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.warmup_factor*config.optim_lr,
        peak_value=config.optim_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.optim_ne * config.trn_steps_per_epoch)
    heavi_scheduler = [
        optax.constant_schedule(0), optax.constant_schedule(1)]
    refer_boundaries = [config.start_cls*config.trn_steps_per_epoch]
    refer_fn = optax.join_schedules(
        schedules=heavi_scheduler,
        boundaries=refer_boundaries
    )
    if config.optim_base == "adam":
        base_optim = partial(
            optax.adamw, learning_rate=scheduler, weight_decay=config.optim_weight_decay
        )
    elif config.optim_base == "sgd":
        base_optim = partial(optax.sgd, learning_rate=scheduler,
                             momentum=config.optim_momentum)
    else:
        raise NotImplementedError
    partition_optimizers = {
        "base": optax.set_to_zero() if not config.train_base else base_optim(),
        "score": base_optim(),
        "cls": optax.set_to_zero(),
        "crt": base_optim()
    }

    def tagging(path, v):
        def cls_list():
            for i in range(config.fat):
                if f"cls_{i}" in path:
                    return True
            return False
        if "base" in path:
            return "base"
        elif "score" in path:
            return "score"
        elif "cls" in path:
            return "cls"
        elif "crt" in path:
            return "crt"
        elif cls_list():
            return "cls"
        else:
            raise NotImplementedError
    partitions = flax.core.freeze(
        flax.traverse_util.path_aware_map(tagging, variables["params"]))
    optimizer = optax.multi_transform(partition_optimizers, partitions)

    # ------------------------------------------------------------------------
    # create train state
    # ------------------------------------------------------------------------
    state_rng, sub_rng = jax.random.split(sub_rng)
    
    class TrainState(train_state.TrainState):
        batch_stats: Any
        rng: Any
        ema_params: Any
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
    @jax.jit
    def mse_loss(noise, output):
        p = config.mse_power
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.abs(noise-output)**p, axis=sum_axis)
        return loss

    @jax.jit
    def ce_loss(logits, labels):
        target = common_utils.onehot(labels, num_classes=logits.shape[-1])
        pred = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target*pred, axis=-1)
        return loss

    @jax.jit
    def kld_loss(target_list, refer_list, T=1.):
        if not isinstance(target_list, list):
            target_list = [target_list]
        if not isinstance(refer_list, list):
            refer_list = [refer_list]
        kld_sum = 0
        count = 0
        for tar, ref in zip(target_list, refer_list):
            assert len(tar.shape) == 2, f"{tar.shape}"
            assert len(ref.shape) == 2, f"{ref.shape}"
            logq = jax.nn.log_softmax(tar/T, axis=-1)
            logp = jax.nn.log_softmax(ref/T, axis=-1)
            q = jnp.exp(logq)
            integrand = q*(logq-logp)
            kld = jnp.sum(integrand, axis=-1)
            kld_sum += T**2*kld
            count += 1
        return kld_sum/count

    @jax.jit
    def self_kld_loss(target_list, T=1.):
        N = len(target_list)
        assert N > 1
        kld_sum = 0
        count = N*(N-1)
        logprob_list = [jax.nn.log_softmax(
            tar/T, axis=-1) for tar in target_list]
        for logq in logprob_list:
            for logp in logprob_list:
                q = jnp.exp(logq)
                integrand = q*(logq-logp)
                kld = jnp.sum(integrand, axis=-1)
                kld_sum += T**2*kld
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

    def kld_loss_fn(t, r, marker):
        return reduce_sum(
            kld_loss(t, r), marker)

    def self_kld_loss_fn(t, marker):
        return reduce_sum(self_kld_loss(t), marker)

    # ------------------------------------------------------------------------
    # define step collecting features and logits
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_label(batch):
        z0_list = []
        logitsA = []
        zB, logitsB = forward_resnet(resnet_param_list[0], batch["images"])
        for i, res_params_dict in enumerate(resnet_param_list):
            if batch.get("images_tar") is None:
                images = batch["images"]
            else:
                images = batch["images_tar"]
            z0, logits0 = forward_resnet(res_params_dict, images)
            z0_list.append(z0)
            # logits0: (B, d)
            logits0 = logits0 - logits0.mean(-1, keepdims=True)
            logitsA.append(logits0)
        zA = jnp.concatenate(z0_list, axis=-1)
        if config.ensemble_prediction:
            if config.ensemble_exclude_a:
                _logitsA = logitsA[1:]
            else:
                _logitsA = logitsA
            logprobs = jnp.stack([jax.nn.log_softmax(l) for l in _logitsA])
            ens_logprobs = jax.scipy.special.logsumexp(
                logprobs, axis=0) - np.log(logprobs.shape[0])
            ens_logits = ens_logprobs - ens_logprobs.mean(-1, keepdims=True)
            logitsA = [ens_logits]
        batch["zB"] = zB
        batch["zA"] = zA
        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        return batch

    # ------------------------------------------------------------------------
    # define step sampling mode out of ensemble members
    # ------------------------------------------------------------------------
    def replace_base_cls(base_params, cls_params, state):
        params = state.params.unfreeze()
        ema_params = state.ema_params.unfreeze()
        batch_stats = state.batch_stats
        base_params, base_batch = base_params
        cls_params, cls_batch = cls_params

        params["base"] = base_params
        ema_params["base"] = base_params
        if batch_stats.get("base") is not None:
            batch_stats["base"] = base_batch

        params["cls"] = cls_params
        ema_params["cls"] = cls_params
        if batch_stats.get("cls") is not None:
            batch_stats["cls"] = cls_batch
        return state.replace(
            params=freeze(params),
            ema_params=freeze(ema_params),
            batch_stats=batch_stats
        )

    def step_mode(state):
        state = jax_utils.unreplicate(state)
        _, idx_rng = jax.random.split(state.rng)
        N = len(resnet_param_list)
        idx = jax.random.randint(idx_rng, (), 0, N)
        state = replace_base_cls(
            base_params_list[idx], cls_params_list[idx], state)
        state = jax_utils.replicate(state)
        return state

    def step_modeA(state):
        state = jax_utils.unreplicate(state)
        state = replace_base_cls(
            base_params_list[0], cls_params_list[0], state)
        state = jax_utils.replicate(state)
        return state

    # ------------------------------------------------------------------------
    # define train step
    # ------------------------------------------------------------------------

    def loss_func(params, state, batch, train=True):
        logitsA = batch["logitsA"]
        logitsA = [l for l in batch["logitsA"]]  # length fat list
        if config.residual_prediction:
            logitsB = batch["logitsB"]
            _logitsA = [lA-logitsB for lA in logitsA]
            _logitsA = jnp.concatenate(
                _logitsA, axis=-1)  # (B, fat*num_classes)
        else:
            _logitsA = jnp.concatenate(
                logitsA, axis=-1)  # (B, fat*num_classes)
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        output = state.apply_fn(
            params_dict, score_rng,
            _logitsA, batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        (
            epsilon, l_t, t, mu_t, sigma_t
        ), logits0eps = output[0] if train else output

        _sigma_t = expand_to_broadcast(sigma_t, l_t, axis=1)
        diff = (l_t-_logitsA) / _sigma_t
        score_loss = mse_loss(epsilon, diff)
        if config.prob_loss:
            pA = jnp.concatenate([jax.nn.softmax(l, axis=-1)
                                 for l in logitsA], axis=-1)
            p0eps = jax.nn.softmax(logits0eps, axis=-1)
            p0eps = rearrange(p0eps, "n b d -> b (n d)")
            _sigma_t = expand_to_broadcast(sigma_t, p0eps, axis=1)
            score_loss += mse_loss(p0eps/_sigma_t, pA/_sigma_t)
        score_loss = reduce_mean(score_loss, batch["marker"])
        if config.kld_joint:
            if config.residual_prediction:
                logits0eps = [logitsB + l for l in logits0eps]
            else:
                logits0eps = [l for l in logits0eps]
            cls_loss = kld_loss(
                logits0eps, logitsA, T=config.kld_temp) / sigma_t**2
            cls_loss = reduce_mean(cls_loss, batch["marker"])
        else:
            cls_loss = 0

        total_loss = config.gamma*score_loss + config.beta*cls_loss

        count = jnp.sum(batch["marker"])
        metrics = OrderedDict({
            "loss": total_loss*count,
            "score_loss": score_loss*count,
            "cls_loss": cls_loss*count,
            "count": count,
        })

        return total_loss, (metrics, new_model_state)

    def get_loss_func():
        return loss_func

    @ partial(jax.pmap, axis_name="batch")
    def step_train(state, batch):

        def loss_fn(params):
            _loss_func = get_loss_func()
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
    @ partial(jax.pmap, axis_name="batch")
    def step_valid(state, batch):

        _loss_func = get_loss_func()
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
            if i != 0:  # Not B
                acc = evaluate_acc(
                    logprob, labels, log_input=True, reduction="none")
                nll = evaluate_nll(
                    logprob, labels, log_input=True, reduction='none')
                acc = reduce_sum(acc, marker)
                nll = reduce_sum(nll, marker)
                acc_list.append(acc)
                nll_list.append(nll)
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

    def sample_func(state, batch, steps=(config.T+1)//2):
        logitsB = batch["logitsB"]
        logitsA = batch["logitsA"]
        zB = batch["zB"]
        zA = batch["zA"]
        labels = batch["labels"]
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        _dsb_sample = partial(
            dsb_sample_cont if config.dsb_continuous else dsb_sample,
            config=config, dsb_stats=dsb_stats, z_dsb_stats=z_dsb_stats, steps=steps)
        logitsC, _zC = model_bd.sample(
            score_rng, _dsb_sample, batch["images"])
        logitsC = rearrange(logitsC, "n (t b) z -> t n b z", t=steps+1)
        logitsC_inter = logitsC
        if config.medium:
            last2 = jax.nn.log_softmax(logitsC[-2:], axis=-1)
            last2 = jax.scipy.special.logsumexp(last2, axis=0) - np.log(2)
            logitsC = last2 - last2.mean(-1, keepdims=True)
        else:
            logitsC = logitsC[-1]
        _, B, C = logitsC.shape
        if config.residual_prediction:
            assert config.joint == 2
            logitsC = [logitsB+l for l in logitsC]
        else:
            logitsC = [l for l in logitsC]

        kld = kld_loss_fn(logitsA, logitsC, batch["marker"])
        rkld = kld_loss_fn(logitsC, logitsA, batch["marker"])
        skld = kld_loss_fn(logitsC, [logitsB]*len(logitsC), batch["marker"])
        rskld = kld_loss_fn([logitsB]*len(logitsC), logitsC, batch["marker"])
        (
            acc_list, nll_list, cum_acc_list, cum_nll_list
        ) = ensemble_accnll([logitsB]+logitsC, labels, batch["marker"])
        metrics = OrderedDict({
            "kld": kld, "rkld": rkld,
            "skld": skld, "rskld": rskld,
            "count": jnp.sum(batch["marker"]),
        })
        if config.fat > 1:
            pkld = self_kld_loss_fn(logitsC, batch["marker"])
            metrics["pkld"] = pkld
        if config.fat > 1 or config.ensemble_prediction > 1:
            wC = config.ensemble_prediction if config.ensemble_prediction > 1 else config.fat
            wacc, wnll = weighted_accnll(
                [logitsB], logitsC, 1, wC, labels, batch["marker"])
            metrics["w_ens_acc"] = wacc
            metrics["w_ens_nll"] = wnll

        for i, (acc, nll, ensacc, ensnll) in enumerate(
                zip(acc_list, nll_list, cum_acc_list, cum_nll_list), start=1):
            metrics[f"acc{i}"] = acc
            metrics[f"nll{i}"] = nll
            metrics[f"ens_acc{i}"] = ensacc
            metrics[f"ens_nll{i}"] = ensnll
        if config.print_inter:
            (
                iacc_list, _, icum_acc_list, _
            ) = ensemble_accnll([l.reshape(B, C) for l in logitsC_inter[::-1]], labels, batch["marker"])
            for i, (acc, ensacc) in enumerate(zip(iacc_list, icum_acc_list), start=1):
                metrics[f"acc{i}_inter"] = acc
                metrics[f"ens_acc{i}_inter"] = ensacc
        return metrics

    @ partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        steps = (config.T+1)//2 if config.medium else config.T
        metrics = sample_func(state, batch, steps=steps)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    @partial(jax.pmap, axis_name="batch")
    def step_sample_valid(state, batch):
        steps = config.T
        metrics = sample_func(state, batch, steps=steps)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    @ jax.jit
    def accnll(logits, labels, marker):
        logprob = jax.nn.log_softmax(logits, axis=-1)
        acc = evaluate_acc(
            logprob, labels, log_input=True, reduction="none")
        nll = evaluate_nll(
            logprob, labels, log_input=True, reduction='none')
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        return acc, nll

    @jax.jit
    def weighted_accnll(logitsX, logitsY, wX, wY, labels, marker):
        ensprobX = sum(jnp.exp(jax.nn.log_softmax(lg, axis=-1))
                       for lg in logitsX) / len(logitsX)
        ensprobY = sum(jnp.exp(jax.nn.log_softmax(lg, axis=-1))
                       for lg in logitsY) / len(logitsY)
        ensprob = (wX*ensprobX+wY*ensprobY)/(wX+wY)
        acc = evaluate_acc(
            ensprob, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            ensprob, labels, log_input=False, reduction='none')
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        return acc, nll

    @partial(jax.pmap, axis_name="batch")
    def step_acc_ref(state, batch):
        logitsB = batch["logitsB"]
        logitsA = batch["logitsA"]
        labels = batch["labels"]

        kld = kld_loss_fn([logitsB]*len(logitsA), logitsA, batch["marker"])
        rkld = kld_loss_fn(logitsA, [logitsB]*len(logitsA), batch["marker"])
        acc_from, nll_from = accnll(logitsB, labels, batch["marker"])
        (
            acc_list, nll_list, cum_acc_list, cum_nll_list
        ) = ensemble_accnll([logitsB]+logitsA, labels, batch["marker"])
        metrics = OrderedDict({
            "kld_ref": kld, "rkld_ref": rkld,
            "acc_from": acc_from, "nll_from": nll_from,
            "count": jnp.sum(batch["marker"])
        })
        for i, (acc, nll, ensacc, ensnll) in enumerate(
                zip(acc_list, nll_list, cum_acc_list, cum_nll_list), start=1):
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
        if config.distribution == 1:
            count = jnp.sum(batch["marker"])
            x = batch["images"]
            batch_size = x.shape[0]
            a = config.mixup_alpha
            beta_rng, perm_rng = jax.random.split(state.rng)

            lamda = jax.random.beta(beta_rng, a, a)
            lamda = jnp.where(lamda > 0.5, 1-lamda, lamda)

            perm_x = jax.random.permutation(perm_rng, x)
            mixed_x = (1-lamda)*x+lamda*perm_x
            mixed_x = jnp.where(count == batch_size, mixed_x, x)

            batch["images"] = mixed_x
            batch["images_tar"] = x
        else:
            count = jnp.sum(batch["marker"])
            x = batch["images"]
            y = batch["labels"]
            batch_size = x.shape[0]
            a = config.mixup_alpha
            beta_rng, perm_rng = jax.random.split(state.rng)

            lamda = jnp.where(a > 0, jax.random.beta(beta_rng, a, a), 1)

            perm_x = jax.random.permutation(perm_rng, x)
            perm_y = jax.random.permutation(perm_rng, y)
            mixed_x = (1-lamda)*x+lamda*perm_x
            mixed_y = jnp.where(lamda < 0.5, y, perm_y)
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
    wandb.define_metric(f"val/ens_acc{config.fat}", summary="max")
    wandb.define_metric(f"val/ens_nll{config.fat}", summary="min")
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
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        inter_samples = False
        for k, v in summarized.items():
            if "count" in k:
                continue
            elif "lr" in k:
                continue
            if "_inter" in k:
                inter_samples = True
            summarized[k] /= summarized[f"{key}/count"]
        if inter_samples and key == "tst":
            assert summarized.get(f"tst/acc{config.T+1}_inter") is None
            acc_str = ",".join(
                [f"ACC({config.T+1-i})   {summarized[f'{key}/acc{i}_inter']:.4f}" for i in range(1, config.T+1)])
            ens_acc_str = ",".join(
                [f"ENS({config.T+1}to{config.T+1-i}){summarized[f'{key}/ens_acc{i}_inter']:.4f}" for i in range(1, config.T+1)])
            print_fn(acc_str)
            print_fn(ens_acc_str)
        if inter_samples:
            for i in range(1, config.T+1):
                del summarized[f"{key}/acc{i}_inter"]
                del summarized[f"{key}/ens_acc{i}_inter"]

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
        print_bin = (float("inf"), None)
        for batch_idx, batch in enumerate(train_loader):
            batch_rng = jax.random.fold_in(train_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            if config.mixup_alpha > 0:
                batch = step_mixup(state, batch)
            if config.distribution == 2:
                state = step_mode(state)
            batch = step_label(batch)
            state, metrics = step_train(state, batch)
            if epoch_idx == 0:
                acc_ref_metrics = step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if epoch_idx % 10 == 0:
                acc_metrics = step_sample_valid(state, batch)
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
            if config.distribution == 2:
                state = step_modeA(state)
            batch = step_label(batch)
            metrics = step_valid(state, batch)
            if epoch_idx == 0:
                acc_ref_metrics = step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            acc_metrics = step_sample_valid(state, batch)
            metrics.update(acc_metrics)
            valid_metrics.append(metrics)

        valid_summarized = summarize_metrics(valid_metrics, "val")
        valid_summary.update(valid_summarized)
        wl.log(valid_summary)

        # ------------------------------------------------------------------------
        # test by getting features from each resnet
        # ------------------------------------------------------------------------
        if config.ensemble_prediction:
            criteria = valid_summarized[f"val/acc1"]
        elif config.best_nll:
            criteria = -valid_summarized[f"val/ens_nll{config.fat}"]
        else:
            criteria = valid_summarized[f"val/ens_acc{config.fat}"]
        if best_acc < criteria:
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(test_loader):
                batch_rng = jax.random.fold_in(test_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                if config.distribution == 2:
                    state = step_modeA(state)
                batch = step_label(batch)
                metrics = step_valid(state, batch)
                if best_acc == -float("inf"):
                    acc_ref_metrics = step_acc_ref(state, batch)
                    metrics.update(acc_ref_metrics)
                acc_metrics = step_sample(state, batch)
                metrics.update(acc_metrics)
                test_metrics.append(metrics)

            test_summarized = summarize_metrics(test_metrics, "tst")
            test_summary.update(test_summarized)
            wl.log(test_summary)
            best_acc = criteria
            if config.save:
                assert not config.nowandb
                save_path = os.path.join(config.save, wandb.run.name)
                save_state = jax_utils.unreplicate(state)
                if getattr(config, "dtype", None) is not None:
                    config.dtype = str(config.dtype)
                ckpt = dict(
                    params=save_state.ema_params,
                    batch_stats=save_state.batch_stats,
                    config=vars(config)
                )
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(
                    ckpt_dir=save_path,
                    target=ckpt,
                    step=epoch_idx,
                    overwrite=True,
                    orbax_checkpointer=orbax_checkpointer
                )

        wl.flush()

        jax.random.normal(rng, ()).block_until_ready()
        if jnp.isnan(train_summarized["trn/loss"]):
            print_fn("NaN detected")
            break

    wandb.finish()










def launch_distill(args):
    """
      Distillation mode
    """
    # Distillation from trained model.
    ckpt_dir = args.dist_ckpt
    params, batch_stats, config = load_ckpt(ckpt_dir)

    print_fn(f"Loaded checkpoint from {ckpt_dir} to start distillation.")

    # distillation with K linear steps (default: K = 1)
    distill_steps = args.dist_n_T

    












def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults_dsb.default_argument_parser()

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    parser.add_argument("--model_planes", default=16, type=int)
    parser.add_argument("--model_blocks", default=None, type=str)
    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=2e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--warmup_factor', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
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
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument("--tag", default=None, type=str)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--lgamma", default=1, type=float)
    parser.add_argument("--zgamma", default=1, type=float)
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
    # MIMO classifier
    parser.add_argument("--mimo_cls", default=1, type=int)
    # Ensemble Prediction
    parser.add_argument("--ensemble_prediction", default=0, type=int)
    parser.add_argument("--ensemble_exclude_a", action="store_true")
    # Residual (logitB-logitA) Prediction
    parser.add_argument("--residual_prediction", action="store_true")
    # Depthwise Seperable Conv
    parser.add_argument("--dsc", action="store_true")
    # get intermediate samples
    parser.add_argument("--medium", action="store_true")
    # FiterResponseNorm and Swish activation in Score net
    parser.add_argument("--frn_swish", action="store_true")
    # Score net input scaling (equivalent to NN normalization)
    parser.add_argument("--input_scaling", default=1, type=float)
    # Score net width multiplier
    parser.add_argument("--width_multi", default=1, type=float)
    # print intermediate samples
    parser.add_argument("--print_inter", action="store_true")
    # continuous diffusion
    parser.add_argument("--dsb_continuous", action="store_true")
    # distribution type; 0: 1to1, 1: mixup, 2: mode, 3: annealing
    parser.add_argument("--distribution", default=0, type=int)
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=3e-4, type=float)
    parser.add_argument("--linear_noise", action="store_true")
    # Fat DSB (A+A -> B1+B2)
    parser.add_argument("--fat", default=1, type=int)
    # joint diffusion
    parser.add_argument("--joint_depth", default=1, type=int)
    parser.add_argument("--kld_joint", action="store_true")
    parser.add_argument("--kld_temp", default=1, type=float)
    parser.add_argument("--forget", default=0, type=int)
    parser.add_argument("--prob_loss", action="store_true")
    parser.add_argument("--ce_loss", default=0, type=float)
    parser.add_argument("--mimo_cond", action="store_true")
    parser.add_argument("--start_temp", default=1, type=float)
    # ---------------------------------------------------------------------------------------
    # networks
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--n_feat", default=256, type=int)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--droprate", default=0, type=float)
    parser.add_argument("--feature_name", default="feature.layer3stride2")
    # UNetModel only
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--num_res_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--use_scale_shift_norm", action="store_true")
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--dist_ckpt", default="None", type=str,
                        help='Checkpoint path for distillation')
    parser.add_argument("--dist_n_T", default=1, type=int,
                        help="Number of steps for distillation.")
    # distillation parameters

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    # Check if the distillation path is defined 
    if ('distillation' in args) and args.distillation:
        print_fn("distillation mode")
        distillation = True
        assert args.dist_ckpt != "None"
        # For example: "/mnt/gsai/hyunsu/dsb-bnn/checkpoints/dbn/c10/t235/cosmic-sky-838"
        # TODO: Make sure that the architecture of the restored ckpt corresponds to that of the config file.
    else:
        print_fn("Raw train and eval mode")
        distillation = False

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

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

    if distillation:
        launch_distill(args)
    else:
        launch_train(args)


if __name__ == '__main__':
    main()
