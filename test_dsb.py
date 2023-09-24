from random import choice
from collections import OrderedDict
import argparse
import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import os
import datetime
import optax
from functools import partial
import flax
from flax import jax_utils
from flax.training import checkpoints, common_utils
from tqdm import tqdm
from tabulate import tabulate
from data.build import build_dataloaders
import sys
from dbn import build_dbn, pdict, dsb_sample, get_resnet
from einops import rearrange
from easydict import EasyDict
from giung2.metrics import evaluate_acc, evaluate_nll
import defaults_sgd


def get_config(ckpt_config):
    for k, v in ckpt_config.items():
        if isinstance(v, dict) and v.get("0") is not None:
            l = []
            for k1, v1 in v.items():
                l.append(v1)
            ckpt_config[k] = tuple(l)
    config = EasyDict(ckpt_config)
    model_dtype = getattr(config, "dtype", None) or "float32"
    if "float32" in model_dtype:
        config.dtype = jnp.float32
    elif "float16" in model_dtype:
        config.dtype = jnp.float16
    else:
        raise NotImplementedError
    if getattr(config, "num_classes", None) is None:
        if config.data_name == "CIFAR10_x32":
            config.num_classes = 10
        elif config.data_name == "CIFAR100_x32":
            config.num_classes = 100
        elif config.data_name == "TinyImageNet200_x64":
            config.num_classes = 200
    if getattr(config, "image_stats", None) is None:
        config.image_stats = dict(
            m=jnp.array(defaults_sgd.PIXEL_MEAN),
            s=jnp.array(defaults_sgd.PIXEL_STD))
    if getattr(config, "model_planes", None) is None:
        if config.data_name == "CIFAR10_x32" and config.model_style == "FRN-Swish":
            config.model_planes = 16
            config.model_blocks = None
        elif config.data_name == "CIFAR100_x32" and config.model_style == "FRN-Swish":
            config.model_planes = 16
            config.model_blocks = None
        elif config.data_name == "TinyImageNet200_x64" and config.model_style == "FRN-Swish":
            config.model_planes = 64
            config.model_blocks = "3,4,6,3"
    return config


def launch(args):
    rng = jax.random.PRNGKey(args.seed)
    # ------------------------------------------------------------------------------------------------------------
    # load checkpoint and configuration
    # ------------------------------------------------------------------------------------------------------------

    def load_ckpt(dirname):
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=dirname,
            target=None
        )
        if ckpt.get("model") is not None:
            if ckpt.get("Dense_0") is not None:
                params = ckpt["model"]
                batch_stats = dict()
            else:
                params = ckpt["model"]["params"]
                batch_stats = ckpt["model"]["batch_stats"]
        else:
            params = ckpt["params"]
            batch_stats = ckpt["batch_stats"]
        config = get_config(ckpt["config"])
        return params, batch_stats, config
    if getattr(args, "checkpoints", None) is not None:
        checkpoint_list = args.checkpoints
    else:
        checkpoint_list = [args.checkpoint]
    params, batch_stats, config = load_ckpt(checkpoint_list[0])

    # ------------------------------------------------------------------------------------------------------------
    # define model
    # ------------------------------------------------------------------------------------------------------------
    if args.method == "dbn":
        print("Building Diffusion Bridge Network (DBN)...")
        dbn, (dsb_stats, z_dsb_stats) = build_dbn(config)
    elif args.method in ["naive_ed", "proxy_end2", "de"]:
        print("Building ResNet...")
        resnet = get_resnet(config, head=True)()

    # ------------------------------------------------------------------------------------------------------------
    # load dataset
    # ------------------------------------------------------------------------------------------------------------
    dataloaders = build_dataloaders(config)

    # ------------------------------------------------------------------------------------------------------------
    # define metrics
    # ------------------------------------------------------------------------------------------------------------
    @jax.jit
    def reduce_sum(loss, marker):
        assert len(loss.shape) == 1
        return jnp.where(marker, loss, 0).sum()

    def evaluate_bs(logits, labels, log_input=False, eps=1e-8, reduction="mean"):
        confidences = jax.nn.softmax(logits, axis=-1) if log_input else logits
        targets = common_utils.onehot(labels, num_classes=logits.shape[-1])
        raw_results = jnp.sum((confidences-targets)**2, axis=-1)
        if reduction == "none":
            return raw_results
        elif reduction == "mean":
            return jnp.mean(raw_results)
        elif reduction == "sum":
            return jnp.sum(raw_results)
        else:
            raise NotImplementedError(f"Unkown reduction {reduction}")

    def evaluate_ece_dep(confidences, true_labels, marker, log_input=True, eps=1e-8, reduction="mean", num_bins=15):
        """
        Args:
            confidences (Array): An array with shape [N, K,].
            true_labels (Array): An array with shape [N,].
            log_input (bool): Specifies whether confidences are already given as log values.
            eps (float): Small value to avoid evaluation of log(0) when log_input is False.
            num_bins (int): Specifies the number of bins used by the historgram binning.

        Returns:
            A dictionary of components for expected calibration error.
        """
        log_confidences = confidences if log_input else jnp.log(
            confidences + eps)
        max_confidences = jnp.max(
            jnp.exp(log_confidences) if log_input else confidences, axis=-1)
        max_pred_labels = jnp.argmax(log_confidences, axis=-1)
        raw_accuracies = jnp.equal(max_pred_labels, true_labels)

        bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_frequencies = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = jnp.logical_and(
                max_confidences > bin_lower, max_confidences <= bin_upper)
            in_bin = jnp.where(marker, in_bin, 0)
            sum_in_bin = jnp.sum(in_bin)
            bin_frequencies.append(sum_in_bin)
            mean_raw_acc_in_bin = jnp.sum(
                jnp.where(in_bin, raw_accuracies, 0))/sum_in_bin
            mean_max_conf_in_bin = jnp.sum(
                jnp.where(in_bin, max_confidences, 0))/sum_in_bin
            bin_accuracies.append(mean_raw_acc_in_bin)
            bin_confidences.append(mean_max_conf_in_bin)

        bin_accuracies = jnp.array(bin_accuracies)
        bin_confidences = jnp.array(bin_confidences)
        bin_frequencies = jnp.array(bin_frequencies)
        return jnp.nansum(
            jnp.abs(
                bin_accuracies - bin_confidences
            ) * bin_frequencies / jnp.sum(bin_frequencies)
        )

    def evaluate_ece(confidences, true_labels, marker, log_input=True, eps=1e-8, reduction="mean", num_bins=15):
        """
        Args:
            confidences (Array): An array with shape [N, K,].
            true_labels (Array): An array with shape [N,].
            log_input (bool): Specifies whether confidences are already given as log values.
            eps (float): Small value to avoid evaluation of log(0) when log_input is False.
            num_bins (int): Specifies the number of bins used by the historgram binning.

        Returns:
            A dictionary of components for expected calibration error.
        """
        log_confidences = confidences if log_input else jnp.log(
            confidences + eps)
        max_confidences = jnp.max(
            jnp.exp(log_confidences) if log_input else confidences, axis=-1)
        max_pred_labels = jnp.argmax(log_confidences, axis=-1)
        raw_accuracies = jnp.equal(max_pred_labels, true_labels)

        bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        count = jnp.sum(marker)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = jnp.logical_and(
                max_confidences > bin_lower, max_confidences <= bin_upper)
            in_bin = jnp.where(marker, in_bin, 0)
            sum_in_bin = jnp.sum(in_bin)
            prop_in_bin = sum_in_bin/count
            mean_raw_acc_in_bin = jnp.sum(
                jnp.where(in_bin, raw_accuracies, 0))/sum_in_bin
            mean_max_conf_in_bin = jnp.sum(
                jnp.where(in_bin, max_confidences, 0))/sum_in_bin
            _ece = jnp.abs(mean_max_conf_in_bin -
                           mean_raw_acc_in_bin)*prop_in_bin
            ece += jnp.where(sum_in_bin == 0, 0, _ece)

        return ece*count

    # ------------------------------------------------------------------------------------------------------------
    # define testing procedure
    # ------------------------------------------------------------------------------------------------------------
    @jax.jit
    def test_dbn(state, batch):
        if config.medium:
            steps = (config.T+1)//2
        else:
            steps = config.T
        drop_rng, score_rng = jax.random.split(state["rng"])
        params_dict = pdict(
            params=state["params"],
            image_stats=config.image_stats,
            batch_stats=state["batch_stats"],
        )
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        _dsb_sample = partial(
            dsb_sample, config=config, dsb_stats=dsb_stats, z_dsb_stats=z_dsb_stats, steps=steps)
        logitsC, logitsB = model_bd.sample(
            score_rng, _dsb_sample, batch["images"])
        logitsC = rearrange(logitsC, "n (t b) z -> t n b z", t=steps+1)
        if config.medium:
            last2 = jax.nn.log_softmax(logitsC[-2:], axis=-1)
            last2 = jax.scipy.special.logsumexp(last2, axis=0) - np.log(2)
            logitsC = last2 - last2.mean(-1, keepdims=True)
        else:
            logitsC = logitsC[-1]
        logprobsC = jax.nn.log_softmax(logitsC, axis=-1)
        probsC = jnp.exp(logprobsC)
        ens_probsC = jnp.mean(probsC, axis=0)
        return ens_probsC

    @jax.jit
    def test_ens_dbn(state, batch):
        sum_probs = 0
        for m in range(args.ensemble):
            if config.medium:
                steps = (config.T+1)//2
            else:
                steps = config.T
            drop_rng, score_rng = jax.random.split(state["rng"])
            params_dict = pdict(
                params=state["params"][m],
                image_stats=config.image_stats,
                batch_stats=state["batch_stats"][m],
            )
            rngs_dict = dict(dropout=drop_rng)
            model_bd = dbn.bind(params_dict, rngs=rngs_dict)
            _dsb_sample = partial(
                dsb_sample, config=config, dsb_stats=dsb_stats, z_dsb_stats=z_dsb_stats, steps=steps)
            logitsC, logitsB = model_bd.sample(
                score_rng, _dsb_sample, batch["images"])
            logitsC = rearrange(logitsC, "n (t b) z -> t n b z", t=steps+1)
            if config.medium:
                last2 = jax.nn.log_softmax(logitsC[-2:], axis=-1)
                last2 = jax.scipy.special.logsumexp(last2, axis=0) - np.log(2)
                logitsC = last2 - last2.mean(-1, keepdims=True)
            else:
                logitsC = logitsC[-1]
            logprobsC = jax.nn.log_softmax(logitsC, axis=-1)
            probsC = jnp.exp(logprobsC)
            ens_probsC = jnp.mean(probsC, axis=0)
            sum_probs += ens_probsC
        return sum_probs/args.ensemble

    @jax.jit
    def test_naive_ed(state, batch):
        params_dict = dict(params=state["params"])
        mutable = ["intermediates"]
        if getattr(config, "image_stats", None) is not None:
            params_dict["image_stats"] = config.image_stats
        if getattr(config, "batch_stats", None) is not None:
            params_dict["batch_stats"] = state["batch_stats"]
            mutable.append("batch_stats")

        _, new_model_state = resnet.apply(
            params_dict, batch['images'],
            rngs=None,
            mutable=mutable,
            use_running_average=False)
        # cls loss
        logits = new_model_state['intermediates']['cls.logit'][0]
        predictions = jax.nn.log_softmax(logits, axis=-1)
        return jnp.exp(predictions)

    @jax.jit
    def test_de(state, batch):
        sum_probs = 0
        for m in range(args.ensemble):
            params_dict = dict(params=state["params"][m])
            mutable = ["intermediates"]
            if getattr(config, "image_stats", None) is not None:
                params_dict["image_stats"] = config.image_stats
            if getattr(config, "batch_stats", None) is not None:
                params_dict["batch_stats"] = state["batch_stats"][m]
                mutable.append("batch_stats")

            _, new_model_state = resnet.apply(
                params_dict, batch['images'],
                rngs=None,
                mutable=mutable,
                use_running_average=False)
            # cls loss
            logits = new_model_state['intermediates']['cls.logit'][0]
            probs = jax.nn.softmax(logits, axis=-1)
            sum_probs += probs
        return sum_probs/args.ensemble

    @partial(jax.pmap, axis_name="batch")
    def step_test(state, batch):

        if args.method == "dbn":
            ens_probsC = test_dbn(state, batch)
        elif args.method in ["naive_ed", "proxy_end2"]:
            ens_probsC = test_naive_ed(state, batch)
        elif args.method == "de":
            ens_probsC = test_de(state, batch)

        labels = batch["labels"]
        acc = evaluate_acc(
            ens_probsC, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            ens_probsC, labels, log_input=False, reduction="none")
        bs = evaluate_bs(
            ens_probsC, labels, log_input=False, reduction="none")
        marker = batch["marker"]
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        bs = reduce_sum(bs, marker)

        ece = evaluate_ece(
            ens_probsC, labels, marker, log_input=False, reduction="none")
        count = jnp.sum(marker)

        metrics = OrderedDict({
            "acc": acc,
            "nll": nll,
            "bs": bs,
            "ece": ece,
            "count": count
        })
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    # ------------------------------------------------------------------------------------------------------------
    # test model
    # ------------------------------------------------------------------------------------------------------------
    def eval(params, batch_stats, checkpoint_list):
        total_metrics = dict()
        state = dict(
            params=params,
            rng=rng,
            batch_stats=batch_stats
        )
        for idx, dirname in enumerate(checkpoint_list):
            if idx == 0:
                state = jax_utils.replicate(state)
            else:
                params, batch_stats, _ = load_ckpt(dirname)
                state["params"] = jax_utils.replicate(params)
                state["batch_stats"] = jax_utils.replicate(batch_stats)
            sub_rng = jax.random.fold_in(rng, idx)
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                metrics = step_test(state, batch)
                test_metrics.append(metrics)
            test_metrics = common_utils.get_metrics(test_metrics)
            test_metrics = jax.tree_util.tree_map(
                lambda e: e.sum(0), test_metrics)
            for k, v in test_metrics.items():
                if "count" in k:
                    continue
                test_metrics[k] /= test_metrics["count"]
            del test_metrics["count"]
            for k, v in test_metrics.items():
                if total_metrics.get(k) is None:
                    total_metrics[k] = [v]
                else:
                    total_metrics[k].append(v)
        for k, v in total_metrics.items():
            print(
                f"--------------------{k}--------------------")
            for ele in v:
                print(f"{ele:.5f}")
            arr = jnp.stack(v)
            print("-------")
            print(f"{float(arr.mean()):.5f}")
            print(f"{float(arr.std()):.5f}")

    def eval_de(checkpoint_list):
        total_metrics = dict()
        comb_list = []
        for idx in range(5):
            sub_rng = jax.random.fold_in(rng, idx)
            comb = jax.random.choice(sub_rng, len(
                checkpoint_list), shape=(args.ensemble,), replace=False)
            params_list = []
            stats_list = []
            dir_list = []
            for m in range(args.ensemble):
                params, batch_stats, _ = load_ckpt(checkpoint_list[comb[m]])
                params_list.append(params)
                stats_list.append(batch_stats)
                dir_list.append(checkpoint_list[comb[m]].split("/")[-1])
            comb_list.append(dir_list)
            state = dict(
                params=jax_utils.replicate(params_list),
                batch_stats=jax_utils.replicate(stats_list)
            )
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                metrics = step_test(state, batch)
                test_metrics.append(metrics)
            del state
            test_metrics = common_utils.get_metrics(test_metrics)
            test_metrics = jax.tree_util.tree_map(
                lambda e: e.sum(0), test_metrics)
            for k, v in test_metrics.items():
                if "count" in k:
                    continue
                test_metrics[k] /= test_metrics["count"]
            del test_metrics["count"]
            for k, v in test_metrics.items():
                if total_metrics.get(k) is None:
                    total_metrics[k] = [v]
                else:
                    total_metrics[k].append(v)
        for k, v in total_metrics.items():
            print(
                f"--------------------{k}--------------------")
            for c, ele in enumerate(v):
                print(f"{ele:.5f} {','.join(comb_list[c])}")
            arr = jnp.stack(v)
            print("-------")
            print(f"{float(arr.mean()):.5f}")
            print(f"{float(arr.std()):.5f}")

    if args.method == "de":
        eval_de(checkpoint_list)
    else:
        eval(params, batch_stats, checkpoint_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, "r") as f:
            arg_defaults = yaml.safe_load(f)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--method", default="dbn", type=str)
    if args.config is not None:
        parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )
    print_fn = partial(print, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' +
            os.path.dirname(jaxlib.__file__)),
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

    launch(args)


if __name__ == "__main__":
    main()
