from argparse import ArgumentParser
from selectors import EpollSelector
from xml.dom import minicompat
from flax.training import checkpoints, common_utils
from flax import jax_utils
import orbax
# from giung2.models.resnet import FlaxResNet
from models.resnet import FlaxResNet
from giung2.data.build import build_dataloaders
import defaults_sgd as defaults
from functools import partial
import jax.numpy as jnp
import jax
import math
from easydict import EasyDict
import os
from tqdm import tqdm
from sgd_deprecated import TrainState
from flax.training import dynamic_scale as dynamic_scale_lib
import optax


def tree_vectorize(tree):
    flat_value, treedef = jax.tree_util.tree_flatten(tree)
    hflat_value = []
    shapes = []
    for ele in flat_value:
        # print("vectorize", ele.shape)
        shapes.append(ele.shape)
        hflat_value.append(ele.reshape(-1))
    indices = []
    end = 0
    for ele in hflat_value:
        end += len(ele)
        indices.append(end)
    indices.pop()
    indices = jnp.array(indices)

    vector_value = jnp.concatenate(hflat_value)
    return vector_value, treedef, shapes, indices


def tree_unvectorize(vector, treedef, shapes, indices):
    hflat_value = jnp.split(vector, indices)
    flat_value = []
    for sh, ele in zip(shapes, hflat_value):
        rele = ele.reshape(*sh)
        # print("unvectorize", rele.shape)
        flat_value.append(rele)
    value = jax.tree_util.tree_unflatten(treedef, flat_value)
    return value


def get_checkpoint(ckpt_dir, bs, idx=None):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=None, step=idx)
    config = EasyDict(ckpt["config"])
    config.optim_bs = bs
    dataloaders = build_dataloaders(config)
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16
    if config.data_name == "CIFAR10_x32":
        num_classes = 10
    elif config.data_name == "CIFAR100_x32":
        num_classes = 100
    image_shape = dataloaders["image_shape"]

    _ResNet = partial(
        FlaxResNet,
        depth=config.model_depth,
        widen_factor=config.model_width,
        dtype=model_dtype,
        pixel_mean=defaults.PIXEL_MEAN,
        pixel_std=defaults.PIXEL_STD,
        num_classes=num_classes)

    model = _ResNet()

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(image_shape, model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)
    # define dynamic_scale
    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * dataloaders['trn_steps_per_epoch'])
    optimizer = optax.sgd(
        learning_rate=scheduler,
        momentum=config.optim_momentum)

    # build train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables['image_stats'],
        batch_stats=variables['batch_stats'],
        dynamic_scale=dynamic_scale)
    ckpt = dict(model=state, config=ckpt["config"], best_acc=0)
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=ckpt, step=idx)
    return ckpt, dataloaders


def get_hessian(rng, state, config, dataloaders):
    trn_loader = dataloaders['dataloader'](rng=rng)
    trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)

    @jax.jit
    def eval_step(state, batch, params):
        # forward pass
        _, new_model_state = state.apply_fn({
            'params': params,
            'image_stats': state.image_stats,
            'batch_stats': state.batch_stats,
        }, batch['images'][0],
            rngs=None,
            mutable=['intermediates', 'batch_stats'],
            use_running_average=False)

        def compute_nll(new_model_state, batch):
            # compute neg_log_likelihood
            # [B, K,]
            logits = new_model_state['intermediates']['cls.logit'][0]
            target = common_utils.onehot(
                batch['labels'][0], num_classes=logits.shape[-1])          # [B, K,]
            weights = jax.nn.log_softmax(logits, axis=-1)
            mini_neg_log_likelihood = - \
                jnp.sum(target * weights, axis=-1)  # [B,]
            mini_neg_log_likelihood = jnp.sum(jnp.where(
                batch['marker'][0], mini_neg_log_likelihood, jnp.zeros_like(mini_neg_log_likelihood)))
            return mini_neg_log_likelihood

        mini_neg_log_likelihood = compute_nll(
            new_model_state, batch)
        return state, mini_neg_log_likelihood

    def potential_energy(vector_params, treedef, shapes, indices):
        params = tree_unvectorize(vector_params, treedef, shapes, indices)
        neg_log_likelihood = 0
        for batch_idx, batch in tqdm(enumerate(trn_loader, start=1)):
            _, mini_neg_log_likelihood = eval_step(state, batch, params)
        neg_log_likelihood += mini_neg_log_likelihood

        # compute neg_log_prior
        n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
        neg_log_prior = 0.5 * (
            - n_params * jnp.log(2.0 * math.pi)
            + n_params * jnp.log(config.prior_precision + 1e-8)
            + sum([jnp.sum(e**2) for e in jax.tree_util.tree_leaves(params)]) * config.prior_precision)

        # compute posterior_energy
        posterior_energy = neg_log_likelihood + neg_log_prior
        return posterior_energy

    hessian_fn = jax.hessian(potential_energy)

    return hessian_fn


def get_cov(hessian_fn, vector_params):
    hessian = hessian_fn(vector_params)
    print("hess diagonal", hessian)
    print("number of negative or zero diagonal", jnp.sum(hessian <= 0))
    print("hess", hessian)
    cov = jnp.linalg.inv(hessian + 1e-8*jnp.eye(len(hessian)))
    diag = jnp.diagonal(cov)
    print("cov diagonal", diag)
    print("number of negative or zero diagonal", jnp.sum(diag <= 0))
    print("cov", cov)
    return cov


def sample(rng, vector_params, cov):
    s = jax.random.multivariate_normal(rng, vector_params, cov)
    return s


def save_sample(idx, ckpt, params, max_num):
    # def save_sample(idx, target_ckpt, params):
    # state = target_ckpt["model"]
    # new_state = state.replace(params=params)
    # config = target_ckpt["config"]
    # best_acc = target_ckpt["best_acc"]
    # ckpt = dict(model=new_state, config=config, best_acc=best_acc)
    ckpt["model"].replace(params=params)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(config.save, "laplace"),
                                target=ckpt,
                                step=idx,
                                overwrite=False,
                                keep=max_num,
                                orbax_checkpointer=orbax_checkpointer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--num", default=1, type=int)
    parser.add_argument("--bs", default=4, type=int)
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.dir)
    print(f"Loading the checkpoint: {ckpt_dir}")
    sample_dir = os.path.join(ckpt_dir, "laplace")
    if os.path.exists(sample_dir):
        raise AssertionError(f'already existing args.save = {sample_dir}')
    ckpt, dataloaders = get_checkpoint(ckpt_dir, bs=args.bs)
    acc = ckpt["best_acc"]
    print(f"Best acc.: {acc:.2f}")
    state = ckpt["model"]
    config = EasyDict(ckpt["config"])
    config.prior_precision = 1.
    rng = jax.random.PRNGKey(2023)

    data_rng, rng = jax.random.split(rng)
    print("Calculating Hessian..")
    _hessian_fn = get_hessian(data_rng, state, config, dataloaders)
    vector_params, treedef, shapes, indices = tree_vectorize(state.params)
    def hessian_fn(v): return _hessian_fn(v, treedef, shapes, indices)
    cov = get_cov(hessian_fn, vector_params)
    for i in range(args.num):
        sample_rng, rng = jax.random.split(rng)
        s = sample(sample_rng, vector_params, cov)
        print("s", s[0])
        params = tree_unvectorize(s, treedef, shapes, indices)
        save_sample(i, ckpt, params, max_num=args.num)
        print(f"Sample #{i} is saved")
