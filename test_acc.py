from argparse import ArgumentParser
from laplace_sample import get_checkpoint
from giung2.metrics import evaluate_acc, evaluate_nll
import os
from flax import jax_utils
from flax.training import common_utils
import jax.numpy as jnp
from collections import OrderedDict
import jax


def step_val(state, batch):

    # forward pass
    _, new_model_state = state.apply_fn({
        'params': state.params,
        'image_stats': state.image_stats,
        'batch_stats': state.batch_stats,
    }, batch['images'],
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

    # refine and return metrics
    acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
    nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
    cnt = jnp.sum(batch['marker'])

    metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': cnt})
    metrics = jax.lax.psum(metrics, axis_name='batch')
    return metrics


def evaluate(ckpt, dataloaders):
    state = jax_utils.replicate(ckpt["model"])
    print(state.params["Dense_0"]["kernel"])
    p_step_val = jax.pmap(step_val, axis_name="batch")
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
    return test_acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--idx", default=None, type=int)
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.dir)
    print(f"Loading the checkpoint: {ckpt_dir}")
    ckpt, dataloaders = get_checkpoint(ckpt_dir, args.bs, args.idx)
    acc = evaluate(ckpt, dataloaders)
    print(f"Accuracy: {acc:.5f}")
