# Revised from https://github.com/cs-giung/giung2-dev/blob/main/projects/residual-networks/scripts/SGDM.py
from giung2.metrics import evaluate_acc, evaluate_nll
# from giung2.models.resnet import FlaxResNet
from models.resnet import FlaxResNet
from models.resnet_v2 import (
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200
)
from utils import normalize, unnormalize
from giung2.models.layers import FilterResponseNorm
from giung2.data.build import build_dataloaders
import defaults_sgd as defaults
from tensorflow.io import gfile
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils, train_state, checkpoints
from flax import jax_utils, serialization
import orbax
import flax
import optax
import jaxlib
import jax.numpy as jnp
import jax
from collections import OrderedDict
from typing import Any, NamedTuple
from functools import partial
from tabulate import tabulate
import datetime
import os
import sys
sys.path.append('./')


class TrainStateBatch(train_state.TrainState):
    image_stats: Any
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


class TrainState(train_state.TrainState):
    image_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def launch(config, print_fn):

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    _ResNet = {
        "0": partial(
            FlaxResNet,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=model_dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=dataloaders['num_classes']),  # legacy
        "18": ResNet18,
        "34": ResNet34,
        "50": ResNet50,
        "101": ResNet101,
        "152": ResNet152,
        "200": ResNet200,
    }[config.v2]

    if config.v2 != 0:
        model = _ResNet(
            num_classes=dataloaders["num_classes"],
            dtype=model_dtype
        )
    elif config.model_style == "BN-ReLU":
        model = _ResNet()
    elif config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish
        )
    print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 1))))

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init(
            {'params': key},
            jnp.ones(dataloaders['image_shape'], model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define dynamic_scale
    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders['trn_steps_per_epoch'])
    if config.optim == "sgd":
        optimizer = optax.sgd(
            learning_rate=scheduler,
            momentum=config.optim_momentum)
    elif config.optim == "adam":
        optimizer = optax.adam(learning_rate=scheduler)

    # build train state
    state = TrainStateBatch.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables.get('image_stats'),
        batch_stats=variables.get('batch_stats'),
        dynamic_scale=dynamic_scale)

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #

    def step_trn(state, batch, config, scheduler):

        # define loss function
        def loss_fn(params):
            params_dict = dict(params=params)
            mutable = ["intermediates"]
            if state.image_stats is not None:
                params_dict["image_stats"] = state.image_stats
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")

            kwargs = dict()
            if config.v2 == "0":
                x = batch["images"]
                kwargs["use_running_average"] = False
            else:
                x = normalize(batch["images"])
                kwargs["train"] = True

            _, new_model_state = state.apply_fn(
                params_dict,
                x,
                rngs=None,
                mutable=mutable,
                **kwargs)

            # compute loss
            # [B, K,]
            logits = new_model_state['intermediates']['cls.logit'][0]
            target = common_utils.onehot(
                batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
            loss = -jnp.sum(target * jax.nn.log_softmax(logits,
                            axis=-1), axis=-1)      # [B,]
            loss = jnp.sum(
                jnp.where(batch['marker'], loss, jnp.zeros_like(loss))
            ) / jnp.sum(batch['marker'])

            # log metrics
            metrics = OrderedDict({'loss': loss})
            return loss, (metrics, new_model_state)

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
        grads = jax.tree_util.tree_map(
            lambda g, p: g + config.optim_weight_decay * p, grads, state.params)

        # get auxiliaries
        metrics, new_model_state = aux[1]
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)

        # update train state
        if new_model_state["batch_stats"] is None:
            new_state = state.apply_gradients(grads=grads)
        else:
            new_state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats'])

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.params, state.params),
                dynamic_scale=dynamic_scale)
            metrics['dynamic_scale'] = dynamic_scale.scale

        return new_state, metrics

    def step_val(state, batch):
        params_dict = dict(params=state.params)
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats

        kwargs = dict()
        if config.v2 == "0":
            x = batch["images"]
            kwargs["use_running_average"] = True
        else:
            x = normalize(batch["images"])
            kwargs["train"] = False

        _, new_model_state = state.apply_fn(
            params_dict,
            x,
            rngs=None,
            mutable='intermediates',
            **kwargs)

        # compute metrics
        predictions = jax.nn.log_softmax(
            new_model_state['intermediates']['cls.logit'][0], axis=-1)  # [B, K,]
        acc = evaluate_acc(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_trn = jax.pmap(partial(step_trn, config=config,
                          scheduler=scheduler), axis_name='batch')
    p_step_val = jax.pmap(step_val,
                          axis_name='batch')
    state = jax_utils.replicate(state)
    rng = jax.random.PRNGKey(config.seed)
    best_acc = 0.0
    test_acc = 0.0
    test_nll = float('inf')

    for epoch_idx, _ in enumerate(range(config.optim_ne), start=1):

        log_str = '[Epoch {:5d}/{:5d}] '.format(epoch_idx, config.optim_ne)
        rng, data_rng = jax.random.split(rng)

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=data_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader, start=1):
            state, metrics = p_step_trn(state, batch)
            trn_metric.append(metrics)
        trn_metric = common_utils.get_metrics(trn_metric)
        trn_summarized = {f'trn/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.mean(), trn_metric).items()}
        log_str += ', '.join(f'{k} {v:.3e}' for k, v in trn_summarized.items())

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
        for batch_idx, batch in enumerate(val_loader, start=1):
            metrics = p_step_val(state, batch)
            val_metric.append(metrics)
        val_metric = common_utils.get_metrics(val_metric)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), val_metric).items()}
        val_summarized['val/acc'] /= val_summarized['val/cnt']
        val_summarized['val/nll'] /= val_summarized['val/cnt']
        del val_summarized['val/cnt']
        log_str += ', ' + \
            ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

        # ---------------------------------------------------------------------- #
        # Save
        # ---------------------------------------------------------------------- #
        if config.save and best_acc < val_summarized['val/acc']:

            tst_metric = []
            tst_loader = dataloaders['tst_loader'](rng=None)
            tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
            for batch_idx, batch in enumerate(tst_loader, start=1):
                metrics = p_step_val(state, batch)
                tst_metric.append(metrics)
            tst_metric = common_utils.get_metrics(tst_metric)
            tst_summarized = {
                f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
            test_acc = tst_summarized['tst/acc'] / tst_summarized['tst/cnt']
            test_nll = tst_summarized['tst/nll'] / tst_summarized['tst/cnt']

            log_str += ' (best_acc: {:.3e} -> {:.3e}, test_acc: {:.3e}, test_nll: {:.3e})'.format(
                best_acc, val_summarized['val/acc'], test_acc, test_nll)
            best_acc = val_summarized['val/acc']

            # _state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
            # with gfile.GFile(os.path.join("./checkpoints", config.save+'.ckpt'), 'wb') as fp:
            #     fp.write(serialization.to_bytes(_state))
            save_state = jax_utils.unreplicate(state)
            ckpt = dict(model=save_state, config=vars(
                config), best_acc=best_acc)
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpoints.save_checkpoint(ckpt_dir=config.save,
                                        target=ckpt,
                                        step=epoch_idx,
                                        overwrite=True,
                                        orbax_checkpointer=orbax_checkpointer)

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

    parser.add_argument('--optim_ne', default=500, type=int)
    parser.add_argument('--optim_lr', default=1e-4, type=float)
    parser.add_argument('--optim_momentum', default=0.9, type=float)
    parser.add_argument('--optim_weight_decay', default=0.0, type=float)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--precision', default='fp32', type=str)
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument("--v2", default="0", type=str,
                        choices=["18", "34", "50", "101", "152", "200", "18Local"])

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
