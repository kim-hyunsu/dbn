# generic
from typing import Any
from functools import partial

# jax-related
import jax, optax, flax
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze
from flax import traverse_util
import jax.numpy as jnp

# user-defined
from default_args import defaults_dsb, defaults_sgd
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet, FlaxResNetBase
from utils import model_list
from models.resnet import FlaxResNetClassifier3
from models.i2sb import  ClsUnet, DiffusionBridgeNetwork
from models.bridge import dsb_schedules, CorrectionModel


def get_latentbe_state(config, rng):
    class TrainState(train_state.TrainState):
        image_stats: Any = None
        batch_stats: Any = None

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
    if int(config.latentbe.split("/")[-1]) > 9:
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
            pixel_mean=defaults_dsb.PIXEL_MEAN,
            pixel_std=defaults_dsb.PIXEL_STD,
            num_classes=config.num_classes,
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None
        )

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
    ckpt = ckpt["model"] if ckpt.get("model") is not None else ckpt
    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")
    image_stats = ckpt.get("image_stats")
    return params, batch_stats, image_stats


def get_model_list(config):
    name = config.data_name
    style = config.model_style
    shared = config.shared_head
    sgd_state = getattr(config, "sgd_state", False)
    tag = config.tag
    assert tag is not None, "Specify a certain group of checkpoints (--tag)"
    model_dir_list = model_list(name, style, shared, tag)
    return model_dir_list


def get_classifier(config):
    feature_name = config.feature_name
    num_classes = config.num_classes

    module = FlaxResNetClassifier3

    classifier = partial(
        module,
        depth=config.model_depth,
        widen_factor=config.model_width,
        dtype=config.dtype,
        pixel_mean=defaults_sgd.PIXEL_MEAN,
        pixel_std=defaults_sgd.PIXEL_STD,
        num_classes=num_classes,
        num_planes=config.model_planes,
        num_blocks=tuple(
            int(b) for b in config.model_blocks.split(",")
        ) if config.model_blocks is not None else None,
        feature_name=feature_name,
        mimo=1
    )
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


def get_scorenet(config):
    score_input_dim = config.score_input_dim
    in_channels = score_input_dim[-1]
    num_channels = in_channels//config.z_dim[-1] * config.n_feat

    score_func = partial(
        ClsUnet,
        num_input=config.fat,
        p_dim=config.num_classes,
        z_dim=config.z_dim,
        ch=num_channels // config.fat,
        joint=2,
        depth=config.joint_depth,
        version=config.version,
        droprate=config.droprate,
        input_scaling=config.input_scaling,
        width_multi=config.width_multi
    )

    if config.dsc:
        score_func = score_func
    else:
        score_func = partial(
            score_func,
            dsconv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros
            )
        )
    if config.frn_swish:
        score_func = partial(
            score_func,
            norm=FilterResponseNorm,
            relu=flax.linen.swish
        )
    return score_func


def get_base_cls(variables, resnet_param_list):
    base_list = []
    cls_list = []
    for idx in range(len(resnet_param_list)):
        _variables = load_base_cls(
            variables, resnet_param_list, load_cls=True, base_type=idx)

        base_params = _variables["params"]["base"]
        if _variables["batch_stats"].get("base") is not None:
            base_batch_stats = _variables["batch_stats"]["base"]
        else:
            base_batch_stats = None

        cls_params = _variables["params"]["cls"]
        if _variables["batch_stats"].get("cls") is not None:
            cls_batch_stats = _variables["batch_stats"]["cls"]
        else:
            cls_batch_stats = None
        base_list.append((base_params, base_batch_stats))
        cls_list.append((cls_params, cls_batch_stats))
    return base_list, cls_list


def load_base_cls(variables, resnet_param_list, load_cls=True, base_type="A", mimo=1):
    def sorter(x):
        assert "_" in x
        name, num = x.split("_")
        return (name, int(num))

    var = variables.unfreeze()
    if not isinstance(base_type, str):
        def get(key1, key2):
            return resnet_param_list[base_type][key1][key2]
    elif base_type == "A":
        def get(key1, key2, idx=0):
            return resnet_param_list[idx][key1][key2]
    elif base_type == "AVG":
        def get(key1, key2):
            return jax.tree_util.tree_map(
                lambda *args: sum(args)/len(args),
                *[param[key1][key2] for param in resnet_param_list]
            )
    elif base_type == "BE":
        be_params = resnet_param_list[0]

        def get(key1, key2):
            return be_params[key1][key2]
    else:
        raise NotImplementedError

    resnet_params = resnet_param_list[0]
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
                resnet_cls = get("params", k)
                dbn_cls = var["params"]["cls"][cls_k]
                for key, value in resnet_cls.items():
                    rank = len(value.shape)
                    if dbn_cls[key].shape[0] != value.shape[0]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[mimo]+[1]*(rank-1))
                    elif rank > 1 and dbn_cls[key].shape[1] != value.shape[1]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1]+[mimo]+[1]*(rank-2))
                    elif rank > 2 and dbn_cls[key].shape[2] != value.shape[2]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1, 1]+[mimo]+[1]*(rank-3))
                    elif rank > 3 and dbn_cls[key].shape[3] != value.shape[3]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1, 1, 1]+[mimo]+[1]*(rank-4))
                    else:
                        var["params"]["cls"][cls_k][key] = value
            cls_idx += 1

    isbatchnorm = resnet_params.get("batch_stats")
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
                    var["batch_stats"]["cls"][cls_k] = get(
                        "batch_stats", k)
                cls_idx += 1

    return freeze(var)


def build_dbn(config):
    cls_net = get_classifier(config)
    base_net = get_resnet(config, head=False)
    dsb_stats = dsb_schedules(
        config.beta1, config.beta2, config.T, linear_noise=config.linear_noise, continuous=config.dsb_continuous)
    score_net = get_scorenet(config)
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