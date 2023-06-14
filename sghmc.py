from giung2.metrics import evaluate_acc, evaluate_nll
from flax.core.frozen_dict import freeze
from flax import traverse_util
from giung2.models.layers import FilterResponseNorm
# from giung2.models.resnet import FlaxResNet
from models.resnet import FlaxResNet
from giung2.data.build import build_dataloaders
import defaults_sghmc as defaults
from flax.training import common_utils, train_state, checkpoints
from flax import jax_utils
import flax
import optax
import jaxlib
import jax.numpy as jnp
import jax
from collections import OrderedDict
from typing import Any, NamedTuple, Callable
from functools import partial
from tabulate import tabulate
import datetime
import math
import os
import sys
from easydict import EasyDict
import orbax
from utils import jprint
from sgd_trainstate import get_sgd_state
sys.path.append('./')


def get_sghmc_state_legacy(config, dataloaders, model, variables):
    # define optimizer with scheduler
    num_epochs_per_cycle = config.num_epochs_quiet + config.num_epochs_noisy

    temperature = optax.join_schedules(
        schedules=[optax.constant_schedule(0.0),] + sum([[
            optax.constant_schedule(1.0),
            optax.constant_schedule(0.0),
        ] for iii in range(1, config.num_cycles + 1)], []),
        boundaries=sum([[
            (iii * num_epochs_per_cycle - config.num_epochs_noisy) *
            dataloaders['trn_steps_per_epoch'],
            (iii * num_epochs_per_cycle) * dataloaders['trn_steps_per_epoch'],
        ] for iii in range(1, config.num_cycles + 1)], []))

    scheduler = optax.join_schedules(
        schedules=[
            optax.cosine_decay_schedule(
                init_value=config.optim_lr,
                decay_steps=num_epochs_per_cycle *
                dataloaders['trn_steps_per_epoch'],
            ) for _ in range(1, config.num_cycles + 1)
        ], boundaries=[
            iii * num_epochs_per_cycle * dataloaders['trn_steps_per_epoch']
            for iii in range(1, config.num_cycles + 1)
        ])
    optimizer = sghmc_legacy(
        learning_rate=scheduler,
        alpha=(1.0 - config.optim_momentum))

    # build train state
    state = TrainStateLegacy.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables.get('image_stats'),
        batch_stats=variables.get("batch_stats"))

    return state


def get_sghmc_state(config, dataloaders, model, variables):

    # define optimizer with scheduler
    num_epochs_per_cycle = config.num_epochs_quiet + config.num_epochs_noisy

    temp_schedules = [optax.constant_schedule(0.0),] + sum([[
        optax.constant_schedule(1.0),
        optax.constant_schedule(0.0),
    ] for iii in range(1, config.num_cycles + 1)], [])
    temp_boundaries = sum([[
        (iii * num_epochs_per_cycle - config.num_epochs_noisy) *
        dataloaders['trn_steps_per_epoch'],
        (iii * num_epochs_per_cycle) * dataloaders['trn_steps_per_epoch'],
    ] for iii in range(1, config.num_cycles + 1)], [])
    temperature = optax.join_schedules(
        schedules=temp_schedules,
        boundaries=temp_boundaries)

    scheduler = optax.join_schedules(
        schedules=[
            optax.cosine_decay_schedule(
                init_value=config.optim_lr,
                decay_steps=num_epochs_per_cycle *
                dataloaders['trn_steps_per_epoch'],
            ) for _ in range(1, config.num_cycles + 1)
        ], boundaries=[
            iii * num_epochs_per_cycle * dataloaders['trn_steps_per_epoch']
            for iii in range(1, config.num_cycles + 1)
        ])
    if config.shared_head:
        optimizer = sghmc(
            learning_rate=scheduler,
            # alpha=(1.0 - config.optim_momentum))
            alpha=(1.0 - config.optim_momentum),
            init_temp=temperature(0))
    else:
        optimizer = sghmc_legacy(
            learning_rate=scheduler,
            alpha=(1.0 - config.optim_momentum))

    if config.shared_head:
        # load trained head
        variables = variables.unfreeze()
        shared_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.shared_head,
            target=None
        )
        saved = shared_ckpt["model"]["params"].get("Dense_0")
        if saved is None:
            saved = shared_ckpt["model"]["params"]["head"]
        variables["params"]["Dense_0"] = saved
        variables = freeze(variables)
        # freeze head
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}
        param_partitions = freeze(traverse_util.path_aware_map(
            lambda path, v: "frozen" if "Dense_0" in path else "trainable", variables["params"]))
        optimizer = optax.multi_transform(
            partition_optimizer, param_partitions)

    # build train state
    if config.shared_head:
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer,
            image_stats=variables.get('image_stats'),
            batch_stats=variables.get("batch_stats"),
            temperature=temperature(0),
            multi_transform=True)
    else:
        state = TrainStateLegacy.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer,
            image_stats=variables.get('image_stats'),
            batch_stats=variables.get("batch_stats")
        )

    return state


class SGHMCStateLegacy(NamedTuple):
    count: jnp.array
    rng_key: Any
    momentum: Any


class SGHMCState(NamedTuple):
    count: jnp.array
    rng_key: Any
    momentum: Any
    temperature: float


class TrainState(train_state.TrainState):
    image_stats: Any
    batch_stats: Any
    # dynamic_scale: dynamic_scale_lib.DynamicScale
    temperature: float = 1.0
    multi_transform: bool = False

    def apply_gradients(self, *, grads, **kwargs):
        # TODO
        trainable_opt_state = self.opt_state.inner_states["trainable"].inner_state
        trainable_opt_state = trainable_opt_state._replace(
            temperature=kwargs["temperature"])
        self.opt_state.inner_states["trainable"] = self.opt_state.inner_states["trainable"]._replace(
            inner_state=trainable_opt_state)
        updates, new_opt_state = self.tx.update(
            updates=grads,
            state=self.opt_state,
            params=self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs)


class TrainStateLegacy(train_state.TrainState):
    image_stats: Any
    batch_stats: Any
    # dynamic_scale: dynamic_scale_lib.DynamicScale

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            gradients=grads,
            state=self.opt_state,
            params=self.params,
            temperature=kwargs.pop("temperature", 1.0))
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs)


def sghmc(learning_rate, seed=0, alpha=0.1, init_temp=1.0):
    """
    Optax implementation of the SGHMC and SGLD.

    Args:
        learning_rate : A fixed global scaling factor.
        seed (int) : Seed for the pseudo-random generation process (default: 0).
        alpha (float) : A momentum decay value (default: 0.1)
    """
    def init_fn(params):
        return SGHMCState(
            count=jnp.zeros([], jnp.int32),
            rng_key=jax.random.PRNGKey(seed),
            # momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            temperature=init_temp)

    # def update_fn(updates, state, params=None, temperature=1.0):
    def update_fn(updates, state, params=None):
        gradients = updates
        # if params is not None:
        #     _temperature = params.get("TEMPERATURE")
        #     if _temperature is not None:
        #         temperature = _temperature
        #         params = params.unfreeze()
        #         del params["TEMPERATURE"]
        #         params = freeze(params)
        temperature = state.temperature

        del params
        lr = learning_rate(state.count)

        # generate standard gaussian noise
        numvars = len(jax.tree_util.tree_leaves(gradients))
        treedef = jax.tree_util.tree_structure(gradients)
        allkeys = jax.random.split(state.rng_key, num=numvars+1)
        rng_key = allkeys[0]
        noise = jax.tree_util.tree_map(
            lambda p, k: jax.random.normal(k, shape=p.shape),
            gradients, jax.tree_util.tree_unflatten(treedef, allkeys[1:]))

        # compute the dynamics
        momentum = jax.tree_util.tree_map(
            lambda m, g, z: (1 - alpha) * m - lr * g + z * jnp.sqrt(
                2 * alpha * temperature * lr
            ), state.momentum, gradients, noise)
        updates = momentum

        return updates, SGHMCState(
            count=state.count + 1,
            rng_key=rng_key,
            # momentum=momentum)
            momentum=momentum,
            temperature=temperature)

    return optax.GradientTransformation(init_fn, update_fn)


def sghmc_legacy(learning_rate, seed=0, alpha=0.1):
    """
    Optax implementation of the SGHMC and SGLD.

    Args:
        learning_rate : A fixed global scaling factor.
        seed (int) : Seed for the pseudo-random generation process (default: 0).
        alpha (float) : A momentum decay value (default: 0.1)
    """
    def init_fn(params):
        return SGHMCStateLegacy(
            count=jnp.zeros([], jnp.int32),
            rng_key=jax.random.PRNGKey(seed),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(gradients, state, params=None, temperature=1.0):
        del params
        lr = learning_rate(state.count)

        # generate standard gaussian noise
        numvars = len(jax.tree_util.tree_leaves(gradients))
        treedef = jax.tree_util.tree_structure(gradients)
        allkeys = jax.random.split(state.rng_key, num=numvars+1)
        rng_key = allkeys[0]
        noise = jax.tree_util.tree_map(
            lambda p, k: jax.random.normal(k, shape=p.shape),
            gradients, jax.tree_util.tree_unflatten(treedef, allkeys[1:]))

        # compute the dynamics
        momentum = jax.tree_util.tree_map(
            lambda m, g, z: (1 - alpha) * m - lr * g + z * jnp.sqrt(
                2 * alpha * temperature * lr
            ), state.momentum, gradients, noise)
        updates = momentum

        return updates, SGHMCStateLegacy(
            count=state.count + 1,
            rng_key=rng_key,
            momentum=momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def launch(config, print_fn):

    if config.ckpt:
        temp = config.ckpt
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.ckpt, target=None)
        for key, value in ckpt["config"].items():
            if key.startswith("optim") or key == "seed":
                continue  # ignore
            setattr(config, key, value)
        sgd_config = ckpt["config"]
        config.ckpt = temp
        print(f"Best acc: {ckpt['best_acc']:.3f}")
        print(f"model style: {config.model_style}")

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    _ResNet = partial(
        FlaxResNet,
        depth=config.model_depth,
        widen_factor=config.model_width,
        dtype=model_dtype,
        pixel_mean=defaults.PIXEL_MEAN,
        pixel_std=defaults.PIXEL_STD,
        num_classes=dataloaders['num_classes'])

    if config.model_style == 'BN-ReLU':
        model = _ResNet()
    elif config.model_style == "FRN-Swish":
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
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define dynamic_scale
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        raise NotImplementedError(
            'fp16 training on GPU is currently not available...')

    # define optimizer with scheduler
    num_epochs_per_cycle = config.num_epochs_quiet + config.num_epochs_noisy

    temp_schedules = [optax.constant_schedule(0.0),] + sum([[
        optax.constant_schedule(1.0),
        optax.constant_schedule(0.0),
    ] for iii in range(1, config.num_cycles + 1)], [])
    temp_boundaries = sum([[
        (iii * num_epochs_per_cycle - config.num_epochs_noisy) *
        dataloaders['trn_steps_per_epoch'],
        (iii * num_epochs_per_cycle) * dataloaders['trn_steps_per_epoch'],
    ] for iii in range(1, config.num_cycles + 1)], [])
    temperature = optax.join_schedules(
        schedules=temp_schedules,
        boundaries=temp_boundaries)

    scheduler = optax.join_schedules(
        schedules=[
            optax.cosine_decay_schedule(
                init_value=config.optim_lr,
                decay_steps=num_epochs_per_cycle *
                dataloaders['trn_steps_per_epoch'],
            ) for _ in range(1, config.num_cycles + 1)
        ], boundaries=[
            iii * num_epochs_per_cycle * dataloaders['trn_steps_per_epoch']
            for iii in range(1, config.num_cycles + 1)
        ])
    if config.shared_head:
        optimizer = sghmc(
            learning_rate=scheduler,
            # alpha=(1.0 - config.optim_momentum))
            alpha=(1.0 - config.optim_momentum),
            init_temp=temperature(0))
    else:
        optimizer = sghmc_legacy(
            learning_rate=scheduler,
            alpha=(1.0 - config.optim_momentum))

    if config.shared_head:
        # load trained head
        variables = variables.unfreeze()
        shared_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.shared_head,
            target=None
        )
        saved = shared_ckpt["model"]["params"].get("Dense_0")
        if saved is None:
            saved = shared_ckpt["model"]["params"]["head"]
        variables["params"]["Dense_0"] = saved
        variables = freeze(variables)
        # freeze head
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}
        param_partitions = freeze(traverse_util.path_aware_map(
            lambda path, v: "frozen" if "Dense_0" in path else "trainable", variables["params"]))
        optimizer = optax.multi_transform(
            partition_optimizer, param_partitions)
        # _optimizer = optax.multi_transform(
        #     partition_optimizer, param_partitions)

        # def wrapper(updates, state, params, temperature):
        #     if params is None:
        #         params = dict()
        #     else:
        #         params = params.unfreeze()
        #     params["TEMPERATURE"] = temperature
        #     params = freeze(params)
        #     return _optimizer.update(updates, state, params)

        # # optimizer.update = wrapper
        # optimizer = _optimizer._replace(update=wrapper)

    # build train state
    if config.shared_head:
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer,
            image_stats=variables.get('image_stats'),
            batch_stats=variables.get("batch_stats"),
            temperature=temperature(0),
            multi_transform=True)
    else:
        state = TrainStateLegacy.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer,
            image_stats=variables.get('image_stats'),
            batch_stats=variables.get("batch_stats")
        )

    if config.ckpt:
        sgd_config = EasyDict(ckpt["config"])
        sgd_state = get_sgd_state(sgd_config, dataloaders, model, variables)
        ckpt = dict(model=sgd_state, config=dict(),
                    best_acc=ckpt["best_acc"])
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.ckpt, target=ckpt
        )
        sgd_state = ckpt["model"]

        if sgd_state.batch_stats is not None:
            state = state.replace(
                params=sgd_state.params,
                image_stats=sgd_state.image_stats,
                batch_stats=sgd_state.batch_stats)
        else:
            state = state.replace(
                params=sgd_state.params,
                image_stats=sgd_state.image_stats)
        del ckpt
        del sgd_state
        config.save = os.path.join(config.ckpt, "sghmc")

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #
    # def step_trn(state, batch, config, scheduler, num_data, temperature):
    def step_trn(state, batch, config, scheduler, num_data):
        def loss_fn(params):
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
                mutable=mutable,
                use_runnining_average=False)

            # compute neg_log_likelihood
            # [B, K,]
            logits = new_model_state['intermediates']['cls.logit'][0]
            target = common_utils.onehot(
                batch['labels'], num_classes=logits.shape[-1])          # [B, K,]
            neg_log_likelihood = - \
                jnp.sum(target * jax.nn.log_softmax(logits,
                        axis=-1), axis=-1)  # [B,]
            neg_log_likelihood = num_data * jnp.sum(
                jnp.where(batch['marker'], neg_log_likelihood,
                          jnp.zeros_like(neg_log_likelihood))
            ) / jnp.sum(batch['marker'])

            # compute neg_log_prior
            n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
            neg_log_prior = 0.5 * (
                - n_params * jnp.log(2.0 * math.pi)
                + n_params * jnp.log(config.prior_precision + 1e-8)
                + sum([jnp.sum(e**2) for e in jax.tree_util.tree_leaves(params)]) * config.prior_precision)

            # compute posterior_energy
            posterior_energy = neg_log_likelihood + neg_log_prior

            # log metrics
            metrics = OrderedDict({
                'posterior_energy': posterior_energy,
                'neg_log_likelihood': neg_log_likelihood,
                'neg_log_prior': neg_log_prior,
            })
            # return posterior_energy, metrics
            return posterior_energy, (metrics, new_model_state)

        # compute losses and gradients
        aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # get auxiliaries
        # metrics = aux[1]
        (metrics, new_model_state) = aux[1]
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)
        # metrics['temperature'] = temperature(state.step)
        # metrics['temperature'] = state.temp_fn(state.step)

        temp = temperature(state.step)
        if new_model_state.get("batch_stats") is not None:
            # update train state
            new_state = state.apply_gradients(
                grads=grads, temperature=temp,
                # grads=grads,
                batch_stats=new_model_state["batch_stats"])
        else:
            # update train state
            new_state = state.apply_gradients(
                grads=grads, temperature=temp)
            # grads=grads)
        if config.shared_head:
            metrics["temperature"] = state.temperature
        return new_state, metrics

    def step_val(state, batch):

        params_dict = dict(params=state.params)
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats

        _, new_model_state = state.apply_fn(
            params_dict, batch['images'],
            rngs=None,
            mutable='intermediates',
            use_running_average=True)

        # compute metrics
        predictions = jax.nn.log_softmax(
            new_model_state['intermediates']['cls.logit'][0], axis=-1)  # [B, K,]
        acc = evaluate_acc(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics along with softmax predictions
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics, jnp.exp(predictions)

    # p_step_trn = jax.pmap(partial(step_trn,
    #                               config=config,
    #                               scheduler=scheduler,
    #                               num_data=dataloaders['num_data'],
    #                               temperature=temperature), axis_name='batch')
    p_step_trn = jax.pmap(
        partial(step_trn,
                config=config,
                scheduler=scheduler,
                num_data=dataloaders['num_data']),
        axis_name='batch')
    p_step_val = jax.pmap(step_val, axis_name='batch')
    state = jax_utils.replicate(state)
    rng = jax.random.PRNGKey(config.seed)

    # initialize buffer
    val_loader = dataloaders['val_loader'](rng=None)
    val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
    val_marker = jnp.concatenate(
        [batch['marker'].reshape(-1) for batch in val_loader])  # [N,]

    val_loader = dataloaders['val_loader'](rng=None)
    val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
    val_labels = jnp.concatenate(
        [batch['labels'].reshape(-1) for batch in val_loader])  # [N,]

    tst_loader = dataloaders['tst_loader'](rng=None)
    tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
    tst_marker = jnp.concatenate(
        [batch['marker'].reshape(-1) for batch in tst_loader])  # [N,]

    tst_loader = dataloaders['tst_loader'](rng=None)
    tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
    tst_labels = jnp.concatenate(
        [batch['labels'].reshape(-1) for batch in tst_loader])  # [N,]

    val_ens_predictions = jnp.zeros(
        (val_labels.shape[0], dataloaders['num_classes']))  # [N, K,]
    tst_ens_predictions = jnp.zeros(
        (tst_labels.shape[0], dataloaders['num_classes']))  # [N, K,]

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')

    # Starting Accuracy
    tst_metric = []
    tst_loader = dataloaders['tst_loader'](rng=None)
    tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
    for batch_idx, batch in enumerate(tst_loader, start=1):
        metrics, predictions = p_step_val(state, batch)
        tst_metric.append(metrics)
    tst_metric = common_utils.get_metrics(tst_metric)
    tst_summarized = {
        f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
    test_acc = tst_summarized['tst/acc'] / \
        tst_summarized['tst/cnt']
    print(f"Starting Accuracy: {test_acc:.3f}")

    for cycle_idx, _ in enumerate(range(config.num_cycles), start=1):

        for epoch_idx, _ in enumerate(range(num_epochs_per_cycle), start=1):

            log_str = '[Cycle {:3d}/{:3d}][Epoch {:3d}/{:3d}] '.format(
                cycle_idx, config.num_cycles, epoch_idx, num_epochs_per_cycle)
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
            trn_summarized = {
                f'trn/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.mean(), trn_metric).items()}
            log_str += ', '.join(f'{k} {v: .3e}' for k,
                                 v in trn_summarized.items())

            if state.batch_stats is not None:
                # synchronize batch normalization statistics
                state = state.replace(
                    batch_stats=cross_replica_mean(state.batch_stats))

            # ---------------------------------------------------------------------- #
            # Valid
            # ---------------------------------------------------------------------- #
            val_metric = []
            val_predictions = []
            val_loader = dataloaders['val_loader'](rng=None)
            val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
            for batch_idx, batch in enumerate(val_loader, start=1):
                metrics, predictions = p_step_val(state, batch)
                val_metric.append(metrics)
                val_predictions.append(
                    predictions.reshape(-1, predictions.shape[-1]))
            val_metric = common_utils.get_metrics(val_metric)
            val_summarized = {
                f'val/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), val_metric).items()}
            val_summarized['val/acc'] /= val_summarized['val/cnt']
            val_summarized['val/nll'] /= val_summarized['val/cnt']
            del val_summarized['val/cnt']
            log_str += ', ' + \
                ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

            if epoch_idx % num_epochs_per_cycle == 0:
                val_predictions = jnp.concatenate(val_predictions)
                val_ens_predictions = (
                    (cycle_idx - 1) * val_ens_predictions + val_predictions) / cycle_idx
                acc = evaluate_acc(val_ens_predictions, val_labels,
                                   log_input=False, reduction='none')
                nll = evaluate_nll(val_ens_predictions, val_labels,
                                   log_input=False, reduction='none')
                acc = jnp.sum(jnp.where(val_marker, acc,
                              jnp.zeros_like(acc))) / jnp.sum(val_marker)
                nll = jnp.sum(jnp.where(val_marker, nll,
                              jnp.zeros_like(nll))) / jnp.sum(val_marker)
                log_str += f', val/ens_acc {acc:.3e}, val/ens_nll {nll:.3e}'

            # ---------------------------------------------------------------------- #
            # Save
            # ---------------------------------------------------------------------- #
            if config.save and epoch_idx % num_epochs_per_cycle == 0:

                tst_metric = []
                tst_predictions = []
                tst_loader = dataloaders['tst_loader'](rng=None)
                tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
                for batch_idx, batch in enumerate(tst_loader, start=1):
                    metrics, predictions = p_step_val(state, batch)
                    tst_metric.append(metrics)
                    tst_predictions.append(
                        predictions.reshape(-1, predictions.shape[-1]))
                tst_metric = common_utils.get_metrics(tst_metric)
                tst_summarized = {
                    f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
                test_acc = tst_summarized['tst/acc'] / \
                    tst_summarized['tst/cnt']
                test_nll = tst_summarized['tst/nll'] / \
                    tst_summarized['tst/cnt']

                tst_predictions = jnp.concatenate(tst_predictions)
                tst_ens_predictions = (
                    (cycle_idx - 1) * tst_ens_predictions + tst_predictions) / cycle_idx
                acc = evaluate_acc(tst_ens_predictions, tst_labels,
                                   log_input=False, reduction='none')
                nll = evaluate_nll(tst_ens_predictions, tst_labels,
                                   log_input=False, reduction='none')
                acc = jnp.sum(jnp.where(tst_marker, acc,
                              jnp.zeros_like(acc))) / jnp.sum(tst_marker)
                nll = jnp.sum(jnp.where(tst_marker, nll,
                              jnp.zeros_like(nll))) / jnp.sum(tst_marker)
                log_str += ' (test_acc: {:.3e}, test_nll: {:.3e}, test_ens_acc: {:.3e}, test_ens_nll: {:.3e})'.format(
                    test_acc, test_nll, acc, nll)

                # _state = jax.device_get(
                #     jax.tree_util.tree_map(lambda x: x[0], state))
                # with gfile.GFile(os.path.join(config.save, f'cycle-{cycle_idx:03d}.ckpt'), 'wb') as fp:
                #     fp.write(serialization.to_bytes(_state))
                save_state = jax_utils.unreplicate(state)
                ckpt = dict(model=save_state, config=vars(
                    config), best_acc=test_acc)
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(ckpt_dir=config.save,
                                            target=ckpt,
                                            step=cycle_idx,
                                            overwrite=False,
                                            keep=1000,
                                            orbax_checkpointer=orbax_checkpointer)
            log_str = datetime.datetime.now().strftime(
                '[%Y-%m-%d %H:%M:%S] ') + log_str
            print_fn(log_str)

            # wait until computations are done
            jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
            if jnp.isnan(trn_summarized['trn/posterior_energy']):
                break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_lr', default=1e-8, type=float,
                        help='base learning rate (default: 1e-7)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')

    parser.add_argument('--num_cycles', default=30, type=int,
                        help='the number of cycles for each sample (default: 30)')
    parser.add_argument('--num_epochs_quiet', default=45, type=int,
                        help='the number of epochs for each exploration stage (default: 45)')
    parser.add_argument('--num_epochs_noisy', default=5, type=int,
                        help='the number of epochs for each sampling stage (default: 5)')

    parser.add_argument('--prior_precision', default=1.0, type=float,
                        help='prior precision (default: 1.0)')

    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument("--ckpt", default=None, type=str)

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
