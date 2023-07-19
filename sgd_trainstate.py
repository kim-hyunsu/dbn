from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state, checkpoints
import optax
import jax
from typing import Any
from flax.core.frozen_dict import freeze
from flax import traverse_util


class TrainState(train_state.TrainState):
    image_stats: Any
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


class TrainStateRNG(train_state.TrainState):
    image_stats: Any
    batch_stats: Any
    rng: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def get_sgd_state(config, dataloaders, model, variables):
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
        optimizer = optax.adam(
            learning_rate=scheduler)
    frozen_keys = []
    if getattr(config, "shared_last3", None) is not None:
        assert config.model_style == "FRN-Swish"
        assert config.model_depth == 32
        params = variables.unfreeze()
        shared_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.shared_last3,
            target=None
        )
        for i in range(28, 34):
            for arc in ["Conv", "FilterResponseNorm"]:
                key = f"{arc}_{i}"
                frozen_keys.append(key)
        frozen_keys.append("Dense_0")
        for key in frozen_keys:
            p = shared_ckpt["model"]["params"][key]
            params["params"][key] = p
        variables = freeze(params)
        # freeze head
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}

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
    elif config.shared_head:
        # load trained head
        params = variables.unfreeze()
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.shared_head,
            target=None
        )
        saved = ckpt["model"]["params"].get("Dense_0")
        if saved is None:
            saved = ckpt["model"]["params"]["head"]
            frozen_keys.append("head")
        else:
            frozen_keys.append("Dense_0")
        params["params"]["Dense_0"] = saved
        variables = freeze(params)
        # freeze head
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}
        param_partitions = freeze(traverse_util.path_aware_map(
            lambda path, v: "frozen" if "Dense_0" in path else "trainable", variables["params"]))
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

    return state


def TrainState2TrainStateRNG(state):
    new_state = TrainStateRNG.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=state.tx,
        image_stats=state.image_stats,
        batch_stats=state.batch_stats,
        rng=jax.random.PRNGKey(0),
        dynamic_scale=state.dynamic_scale
    )
    return new_state
