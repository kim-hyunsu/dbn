from audioop import reverse
from re import M
import numpy as np
from sgd_trainstate import TrainState, TrainStateRNG, get_sgd_state
import wandb
import time
from tqdm import tqdm
from giung2.metrics import evaluate_acc, evaluate_nll
# from giung2.models.resnet import FlaxResNet
from models.resnet import FlaxResNet, FlaxResNetBase
from giung2.models.layers import FilterResponseNorm
from data.build import build_dataloaders
import defaults_sgd as defaults
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils, train_state, checkpoints
from flax import jax_utils
import flax
import orbax
import optax
import jaxlib
import jax.numpy as jnp
import jax
from collections import OrderedDict
from typing import Any
from functools import partial
from tabulate import tabulate
import datetime
import os
import sys
from flax.core.frozen_dict import freeze
from flax import traverse_util

from utils import WandbLogger, get_ens_logits, get_single_batch
sys.path.append('./')
np.random.seed(0)


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=model_dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=dataloaders['num_classes'],
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None
        )
        _base = partial(
            FlaxResNetBase,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=model_dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=dataloaders['num_classes'],
            out=config.shared_level)

    if config.model_style == 'BN-ReLU':
        model = _ResNet()
        base = _base()
    elif config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
        base = _base(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    # initialize model

    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model.dtype))
    _, init_rng = jax.random.split(rng)
    variables = initialize_model(init_rng, model)

    # define dynamic_scale
    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.warmup_factor*config.optim_lr,
        peak_value=config.optim_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.optim_ne * dataloaders['trn_steps_per_epoch'])
    if config.optim == "sgd":
        optimizer = optax.sgd(
            learning_rate=scheduler,
            momentum=config.optim_momentum)
    elif config.optim == "adam":
        optimizer = optax.adamw(learning_rate=scheduler,
                                weight_decay=config.optim_weight_decay)

    frozen_keys = []
    if config.shared_checkpoint is not None:
        base_variables = initialize_model(init_rng, base)

        def sorter(x):
            assert "_" in x
            name, num = x.split("_")
            return (name, int(num))
        params = variables.unfreeze()
        res_param_keys = []
        base_param_keys = []
        for k, v in params["params"].items():
            res_param_keys.append(k)
        for k, v in base_variables["params"].items():
            base_param_keys.append(k)
        res_param_keys = sorted(res_param_keys, key=sorter)
        base_param_keys = sorted(base_param_keys, key=sorter)

        shared_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.shared_checkpoint,
            target=None
        )
        for k in res_param_keys:
            if k in base_param_keys:
                continue
            params["params"][k] = shared_ckpt["model"]["params"][k]
            frozen_keys.append(k)
        variables = freeze(params)
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
            lambda path, v: "frozen" if include(frozen_keys, path) else "trainable", variables["params"]))
        optimizer = optax.multi_transform(
            partition_optimizer, param_partitions)

    # build train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables.get('image_stats'),
        batch_stats=variables.get('batch_stats'),
        dynamic_scale=dynamic_scale)

    teachers = []
    for t in config.teachers:
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=t, target=None
        )
        teacher = ckpt["model"]["params"]
        teachers.append(teacher)
    ensemble_num = len(teachers)

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #
    def loss_func(params, state, batch, train=True):
        def pred(params, logits=False):
            params_dict = dict(params=params)
            mutable = ["intermediates"]
            if state.image_stats is not None:
                params_dict["image_stats"] = state.image_stats
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")

            _, new_model_state = state.apply_fn(
                params_dict, batch['images'],
                rngs=None,
                **(dict(mutable=mutable) if train else dict()),
                use_running_average=False)
            if logits:
                return new_model_state["intermediates"]["cls.logit"][0]
            return new_model_state
        # loss_kd
        tau = config.dist_temp
        new_model_state = pred(params)
        logits = new_model_state["intermediates"]["cls.logit"][0]
        alphas = 1.0 + jnp.exp(logits / tau)  # [N, K,]

        teacher_logits = jnp.stack(
            [pred(t, logits=True) for t in teachers])
        teacher_confs = jax.nn.softmax(
            teacher_logits / tau, axis=-1)  # [M, N, K,]
        teacher_confs_mean = jnp.mean(teacher_confs, axis=0)  # [N, K,]
        precision = 0.5 * (teacher_confs.shape[2] - 1) / jnp.sum(
            teacher_confs_mean * (
                jnp.log(teacher_confs_mean) -
                jnp.mean(jnp.log(teacher_confs), axis=0)
            ), axis=1, keepdims=True)  # [N, 1,]
        betas = 1.0 + (teacher_confs_mean * precision)  # [N, K,]

        reconstruction_term = jnp.sum(
            -teacher_confs_mean * (
                jax.lax.digamma(
                    alphas) - jax.lax.digamma(jnp.sum(alphas, axis=1, keepdims=True))
            ), axis=-1)  # [N,]
        # reconstruction_term = jnp.sum(
        #     -teacher_confs_mean * jax.nn.log_softmax(student_logits, axis=-1), axis=-1) # [N,]

        prior_term = ((
            jax.lax.lgamma(jnp.sum(alphas, axis=1)) -
            jnp.sum(jax.lax.lgamma(alphas), axis=1)
            + jnp.sum(jax.lax.lgamma(jnp.ones_like(betas)), axis=1) -
            jax.lax.lgamma(jnp.sum(jnp.ones_like(betas), axis=1))
        ) + jnp.sum(
            (alphas - jnp.ones_like(betas)) * (
                jax.lax.digamma(
                    alphas) - jax.lax.digamma(jnp.sum(alphas, axis=1, keepdims=True))
            ), axis=1
        )) / jnp.sum(betas, axis=1)  # [N,]
        kd_loss = jnp.mean(prior_term + reconstruction_term)

        target = common_utils.onehot(
            batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
        predictions = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target * predictions, axis=-1)      # [B,]
        loss = jnp.sum(
            jnp.where(batch['marker'], loss, jnp.zeros_like(loss))
        ) / jnp.sum(batch['marker'])

        a = config.dist_alpha
        loss = (1-a)*loss + a*kd_loss

        # accuracy
        acc = evaluate_acc(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])
        # log metrics
        metrics = OrderedDict(
            {'loss': loss, "acc": acc, "nll": nll, "cnt": cnt})

        return loss, (metrics, new_model_state)

    @partial(jax.pmap, axis_name="batch")
    def step_trn(state, batch):
        def loss_fn(params):
            return loss_func(params, state, batch)

        # compute losses and gradients
        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            dynamic_scale, is_fin, aux, grads = dynamic_scale.value_and_grad(
                loss_fn, has_aux=True, axis_name='batch')(state.params)
        else:
            aux, grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')

        # weight decay regularization in PyTorch-style
        if config.optim == "sgd":
            grads = jax.tree_util.tree_map(
                lambda g, p: g + config.optim_weight_decay * p, grads, state.params)

        # get auxiliaries
        metrics, new_model_state = aux[1]
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)

        # update train state
        if new_model_state.get("batch_stats") is not None:
            new_state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats'])
        else:
            new_state = state.apply_gradients(grads=grads)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.params, state.params),
                dynamic_scale=dynamic_scale)
            metrics['dynamic_scale'] = dynamic_scale.scale

        return new_state, metrics

    @partial(jax.pmap, axis_name="batch")
    def step_val(state, batch):
        _, (metrics, _) = loss_func(state.params, state, batch)
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_trn = jax.pmap(partial(step_trn, config=config,
                          scheduler=scheduler), axis_name='batch')
    p_step_val = jax.pmap(step_val,
                          axis_name='batch')
    state = jax_utils.replicate(state)
    best_acc = 0.

    wandb.init(
        project="dsb-bnn-sgd",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/nll", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.summary["params"] = sum(
        x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    # params_flatten = flax.traverse_util.flatten_dict(variables["params"])
    # for k, v in params_flatten.items():
    #     print(k, v.shape)
    wl = WandbLogger()

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        for k, v in summarized.items():
            if "cnt" in k:
                continue
            elif "lr" in k:
                continue
            summarized[k] /= summarized[f"{key}/cnt"]
        del summarized[f"{key}/cnt"]
        return summarized

    for epoch_idx in tqdm(range(config.optim_ne)):
        epoch_rng = jax.random.fold_in(rng, epoch_idx)

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=epoch_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader):
            state, metrics = step_trn(state, batch)
            trn_metric.append(metrics)
        trn_summarized = summarize_metrics(trn_metric, "trn")
        wl.log(trn_summarized)

        if state.batch_stats is not None:
            # synchronize batch normalization statistics
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        # ---------------------------------------------------------------------- #
        # Valid
        # ---------------------------------------------------------------------- #
        val_metric = []
        val_loader = dataloaders['val_loader'](rng=None)
        val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        for batch_idx, batch in enumerate(val_loader):
            metrics = step_val(state, batch)
            val_metric.append(metrics)
        val_summarized = summarize_metrics(val_metric, "val")
        wl.log(val_summarized)

        # ---------------------------------------------------------------------- #
        # Save
        # ---------------------------------------------------------------------- #
        test_condition = best_acc < val_summarized["val/acc"]
        if test_condition:
            tst_metric = []
            tst_loader = dataloaders['tst_loader'](rng=None)
            tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
            for batch_idx, batch in enumerate(tst_loader):
                metrics = step_val(state, batch)
                tst_metric.append(metrics)
            tst_summarized = summarize_metrics(tst_metric, "tst")
            wl.log(tst_summarized)
            best_acc = val_summarized["val/acc"]

            if config.save:
                save_state = jax_utils.unreplicate(state)
                ckpt = dict(
                    params=save_state.params,
                    batch_stats=getattr(save_state, "batch_stats", None),
                    image_stats=getattr(save_state, "image_stats", None),
                    config=vars(config),
                    best_acc=best_acc)
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
            print("loss has NaN")
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    config_f = args.config
    if config_f is not None:
        import yaml
        with open(config_f, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    parser.add_argument("--model_planes", default=16, type=int)
    parser.add_argument("--model_blocks", default=None, type=str)
    parser.add_argument('--optim_ne', default=300, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=0.005, type=float,
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.001, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument("--warmup_factor", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--optim", default="sgd", type=str,
                        choices=["sgd", "adam"])
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--shared_head", default="", type=str)
    parser.add_argument("--shared_last3", default=None, type=str)
    parser.add_argument("--shared_checkpoint", default=None, type=str)
    parser.add_argument("--shared_level", default=None, type=str)
    # ---------------------------------------------------------------------
    # Ensemble distillation
    # ---------------------------------------------------------------------
    parser.add_argument("--teach0", default="", type=str)
    parser.add_argument("--teach1", default="", type=str)
    parser.add_argument("--dist_alpha", default=0.1, type=float)
    parser.add_argument("--dist_temp", default=4, type=float)

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

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
