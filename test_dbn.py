from builtins import getattr, isinstance
from collections import OrderedDict
from sklearn.metrics import r2_score
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
from flax.training import common_utils
from tqdm import tqdm
from tabulate import tabulate
from cls_plot import dbn_plot
from data.build import build_dataloaders
import sys
from dbn_tidy import build_dbn, pdict, dsb_sample, get_resnet
from einops import rearrange
from giung2.metrics import evaluate_acc, evaluate_nll, get_optimal_temperature
from utils import load_ckpt, jprint

def _onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)

def _evaluate_nll_3rank(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    true_target = _onehot(true_labels, num_classes=log_confidences.shape[-1])
    if true_target.shape == log_confidences.shape:
        pass
    elif len(true_target.shape) == len(log_confidences.shape)-1:
        true_target = jnp.expand_dims(true_target, axis=1)
    else:
        raise NotImplementedError
    raw_results = -jnp.sum(jnp.where(true_target, true_target * log_confidences, 0.0), axis=-1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')

def _evaluate_acc_3rank(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    pred_labels = jnp.argmax(confidences, axis=-1)
    if true_labels.shape == confidences.shape:
        pass
    elif len(true_labels.shape) == len(confidences.shape)-2:
        true_labels = jnp.expand_dims(true_labels, axis=1)
    else:
        raise NotImplementedError
    raw_results = jnp.equal(pred_labels, true_labels)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')

def launch(args):
    rng = jax.random.PRNGKey(args.seed)
    # ------------------------------------------------------------------------------------------------------------
    # load checkpoint and configuration
    # ------------------------------------------------------------------------------------------------------------
    rn_config = None
    if "plot" in args.method or "inter" in args.method:
        checkpoint_list = args.checkpoints
        params, batch_stats, config = load_ckpt(args.checkpoint)
        _, _, rn_config = load_ckpt(checkpoint_list[0])
    else:
        if getattr(args, "checkpoints", None) is not None:
            checkpoint_list = args.checkpoints
        else:
            checkpoint_list = [args.checkpoint]
        if isinstance(checkpoint_list, dict):
            print("Nested Checkpoints")
            params, batch_stats, config = load_ckpt(
                list(checkpoint_list.values())[0][0])
        else:
            print("List Checkpoints")
            params, batch_stats, config = load_ckpt(checkpoint_list[0])
    print(checkpoint_list)
    if "r2" in args.method:
        _, _, rn_config = load_ckpt(args.target_checkpoints[0])
    if rn_config is None:
        rn_config = config

    # ------------------------------------------------------------------------------------------------------------
    # define model
    # ------------------------------------------------------------------------------------------------------------
    if args.method == "dbn":
        print("Building Diffusion Bridge Network (DBN)...")
        dbn, (dsb_stats, z_dsb_stats) = build_dbn(config)
    elif args.method in ["naive_ed", "proxy_end2", "de", "mixup-shift"]:
        print("Building ResNet...")
        resnet = get_resnet(config, head=True)()
    elif args.method in ["dbn-plot","dbn-r2","dbn-inter"]:
        print("Building DBN and ResNet...")
        dbn, (dsb_stats, z_dsb_stats) = build_dbn(config)
        resnet = get_resnet(rn_config, head=True)()
    else:
        raise Exception("Invalid method")

    # ------------------------------------------------------------------------------------------------------------
    # load dataset
    # ------------------------------------------------------------------------------------------------------------
    dataloaders = build_dataloaders(config, corrupted=args.corrupted)
    if getattr(args, "mean", None) is None:
        args.mean = jnp.zeros((dataloaders["num_classes"],))
    else:
        assert len(args.mean) == dataloaders["num_classes"]
        args.mean = jnp.array(args.mean)

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

        # return ece*count
        return ece

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
        sum_probs = []
        log_probs = []
        for m in range(args.ensemble):
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
            logitsC = logitsC[-1]
            ####
            # logitsC = logitsC - logitsB/5
            ####
            logprobsC = jax.nn.log_softmax(logitsC, axis=-1)
            log_probs.append(logprobsC)
            probsC = jnp.exp(logprobsC)
            ens_probsC = jnp.mean(probsC, axis=0)
            sum_probs.append(ens_probsC)
            probsB = jax.nn.softmax(logitsB, axis=-1)
            log_probsB = jax.nn.log_softmax(logitsB, axis=-1)
        A = sum(sum_probs) / len(sum_probs)

        # M = args.ensemble
        # N = config.ensemble_prediction
        # A = sum(sum_probs)
        # A -= 0.5*(M-1)*probsB/N
        # A *= N
        # A /= M*(N-1)+1
        # A = jnp.maximum(A, 1e-12)
        # A /= A.sum(-1, keepdims=True)

        # A = (sum(sum_probs)+probsB) / (len(sum_probs)+1)

        # A = sum(log_probs) / len(log_probs)
        # A = jax.nn.softmax(A, axis=-1)[0]

        out = A
        return out

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
                params_dict["image_stats"] = rn_config.image_stats
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
    def step_sample(state, batch):
        if args.method in ["dbn", "dbn-r2"]:
            ens_probsC = test_ens_dbn(state, batch)
        elif args.method in ["naive_ed", "proxy_end2"]:
            ens_probsC = test_naive_ed(state, batch)
        elif args.method == "de":
            ens_probsC = test_de(state, batch)
        return ens_probsC

    @partial(jax.pmap, axis_name="batch")
    def step_mixup_sample(state, batch):
        if isinstance(state["params"], list):
            ens_probsC = test_de(state, batch)
        else:
            ens_probsC = test_naive_ed(state, batch)
        # ens_probsC = jax.nn.softmax(jnp.log(ens_probsC)/1.3, axis=-1)
        return ens_probsC

    @partial(jax.pmap, axis_name="batch")
    def step_annealed_sample(state, batch):
        if isinstance(state["params"], list):
            ens_probsC = test_de(state, batch)
        else:
            ens_probsC = test_naive_ed(state, batch)
        temp = 2+0.4*(1/6)
        ens_probsC = jax.nn.softmax(jnp.log(ens_probsC)/temp, axis=-1)
        return ens_probsC

    @partial(jax.pmap, axis_name="batch")
    def step_mixup(state, batch):
        count = jnp.sum(batch["marker"])
        x = batch["images"]
        y = batch["labels"]
        batch_size = x.shape[0]
        a = state["alpha"]
        beta_rng, perm_rng = jax.random.split(state["rng"])

        lamda = jnp.where(a > 0, jax.random.beta(beta_rng, a, a), 1)
        lamda = jnp.where(lamda < 0.5, lamda, 1-lamda)

        perm_x = jax.random.permutation(perm_rng, x)
        perm_y = jax.random.permutation(perm_rng, y)
        mixed_x = (1-lamda)*x+lamda*perm_x
        mixed_y = jnp.where(lamda < 0.5, y, perm_y)
        mixed_x = jnp.where(count == batch_size, mixed_x, x)
        mixed_y = jnp.where(count == batch_size, mixed_y, y)

        batch["images"] = mixed_x
        batch["labels"] = mixed_y
        return batch

    @partial(jax.pmap, axis_name="batch")
    def step_test(state, batch, optimal_temp):
        def R2score(y, y_hat):
            # y: (B, d)
            y_bar = args.mean[None, :]
            SSE = (y_hat-y_bar)**2
            SST = (y-y_bar)**2
            return SSE, SST

        if args.method in ["dbn", "dbn-r2"]:
            ens_probsC = test_ens_dbn(state, batch)
        elif args.method in ["naive_ed", "proxy_end2"]:
            ens_probsC = test_naive_ed(state, batch)
        elif args.method == "de":
            ens_probsC = test_de(state, batch)

        if "r2" in args.method:
            marker = batch["marker"]
            count = jnp.sum(marker)
            SSE, SST = R2score(ens_probsC, batch["target"])
            SSE = jnp.sum(jnp.where(marker[:, None], SSE, 0), axis=0)
            SST = jnp.sum(jnp.where(marker[:, None], SST, 0), axis=0)

            Mean = jnp.sum(jnp.where(marker[:, None], ens_probsC, 0), axis=0)

            metrics = OrderedDict({
                "SSE": SSE,
                "SST": SST,
                "count": count,
                "mean": Mean
            })
            metrics = jax.lax.psum(metrics, axis_name="batch")
            return metrics
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
        # ece = evaluate_ece(
        #     ens_probsC, labels, marker, log_input=False, reduction="none")

        cal_ens_probsC = jax.nn.softmax(
            jnp.log(ens_probsC)/optimal_temp, axis=-1)
        cnll = evaluate_nll(
            cal_ens_probsC, labels, log_input=False, reduction="none")
        cbs = evaluate_bs(
            cal_ens_probsC, labels, log_input=False, reduction="none")
        cnll = reduce_sum(cnll, marker)
        cbs = reduce_sum(cbs, marker)
        # cece = evaluate_ece(
        #     cal_ens_probsC, labels, marker, log_input=False, reduction="none")

        count = jnp.sum(marker)

        metrics = OrderedDict({
            "acc": acc,
            "nll": nll,
            "bs": bs,
            # "ece": ece,
            "cnll": cnll,
            "cbs": cbs,
            # "cece": cece,
            "count": count
        })
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    @partial(jax.pmap, axis_name="batch")
    def get_target(state, batch):
        ens_probsC = test_de(state, batch)
        batch["target"] = ens_probsC
        return batch

    @partial(jax.pmap, axis_name="batch")
    def get_sample(state, batch):
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
        logitsC = rearrange(logitsC, "n (t b) z -> (n b) t z", t=steps+1)
        probsC = jax.nn.softmax(logitsC, axis=-1)
        return probsC
    
    # ------------------------------------------------------------------------------------------------------------
    # test model
    # ------------------------------------------------------------------------------------------------------------
    def get_batch_only(samples, batch):
        assert len(samples.shape) == 3
        assert len(batch["labels"].shape) == 2
        samples = samples.reshape(-1, samples.shape[-1])
        labels = batch["labels"].reshape(-1)
        marker = batch["marker"].reshape(-1)
        return samples[marker], labels[marker]

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
        arr = []
        for v in total_metrics.values():
            arr.append(v)
        arr = jnp.array(arr)
        arr = rearrange(arr, "m v -> v m")
        for a in arr:
            print(" ".join([f"{ele:.5f}" for ele in a]))

    def eval_mixupshift(params, batch_stats, checkpoint_list):
        source_state = dict(
            params=jax_utils.replicate(params),
            rng=rng,
            batch_stats=jax_utils.replicate(batch_stats)
        )
        params_list = [params]
        stats_list = [batch_stats]
        for idx, dirname in enumerate(checkpoint_list):
            if idx == 0:
                continue
            else:
                params, batch_stats, _ = load_ckpt(dirname)
                params_list.append(params)
                stats_list.append(batch_stats)
        target_state = dict(
            params=jax_utils.replicate(params_list),
            batch_stats=jax_utils.replicate(stats_list)
        )
        def get_samples_and_labels(state, mixup_alpha=0, temp=1):
            train_loader = dataloaders["trn_loader"](rng=None)
            train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
            all_samples = []
            all_labels = []
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                batch_rng = jax.random.fold_in(rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                if mixup_alpha > 0:
                    state["alpha"] = jax_utils.replicate(mixup_alpha)
                    batch = step_mixup(state, batch)
                # if temp > 1:
                #     state["temp"] = jax_utils.replicate(temp)
                state["temp"] = jax_utils.replicate(temp)
                if temp > 1:
                    samples = step_annealed_sample(state, batch)
                else:
                    samples = step_mixup_sample(state, batch)
                _samples, _labels = get_batch_only(samples, batch)
                all_samples.append(_samples)
                all_labels.append(_labels)
            all_samples = jnp.concatenate(all_samples, axis=0)
            all_labels = jnp.concatenate(all_labels, axis=0)
            return all_samples
        def evaluate_KL(q, p):
            assert len(q.shape) == 2
            assert len(p.shape) == 2
            logq = jnp.log(q)
            logp = jnp.log(p)
            integrand = q*(logq-logp)
            kld = jnp.sum(integrand, axis=-1)
            kld = jnp.mean(kld)
            return kld
        def evaluate_MMD(q, p):
            def k(x1, x2):
                sigma = 1
                return jnp.exp(-((x1-x2)**2).sum(-1)/2/sigma**2)


        tar  = get_samples_and_labels(target_state)
        source_list = []
        s = get_samples_and_labels(source_state, mixup_alpha=0.1)
        source_list.append(s)
        s = get_samples_and_labels(source_state, mixup_alpha=0.3)
        source_list.append(s)
        s = get_samples_and_labels(source_state, mixup_alpha=0.5)
        source_list.append(s)
        s = get_samples_and_labels(source_state, mixup_alpha=0.7)
        source_list.append(s)
        s = get_samples_and_labels(source_state, mixup_alpha=0.9)
        source_list.append(s)
        s = get_samples_and_labels(source_state, temp=2)
        source_list.append(s)
        for s in source_list:
            kl = evaluate_KL(s, tar)
            rkl = evaluate_KL(tar, s)
            # mmd = evaluate_MMD(s, tar)
            # print(f"KL {kl:.4f} rKL {rkl:.4f} MMd {mmd:.4f}")
            print(f"KL {kl:.4f} rKL {rkl:.4f}")


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
            # get temperature
            valid_loader = dataloaders["val_loader"](rng=None)
            valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
            all_samples = []
            all_labels = []
            for batch_idx, batch in enumerate(tqdm(valid_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                samples = step_sample(state, batch)
                _samples, _labels = get_batch_only(samples, batch)
                all_samples.append(_samples)
                all_labels.append(_labels)
            all_samples = jnp.concatenate(all_samples, axis=0)
            all_labels = jnp.concatenate(all_labels, axis=0)
            optimal_temp = get_optimal_temperature(all_samples, all_labels, log_input=False)
            optimal_temp = jax_utils.replicate(optimal_temp)
            # test
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            all_samples = []
            all_labels = []
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                metrics = step_test(state, batch, optimal_temp)
                test_metrics.append(metrics)

                samples = step_sample(state, batch)
                _samples, _labels = get_batch_only(samples, batch)
                all_samples.append(_samples)
                all_labels.append(_labels)
            all_samples = jnp.concatenate(all_samples, axis=0)
            all_labels = jnp.concatenate(all_labels, axis=0)
            all_cal_samples = jax.nn.softmax(
                jnp.log(all_samples)/jax_utils.unreplicate(optimal_temp),
                axis=-1
            )
            ece = evaluate_ece(
                all_samples,
                all_labels,
                jnp.ones_like(all_labels, dtype=bool),
                log_input=False,
                reduction="mean"
            )
            cece = evaluate_ece(
                all_cal_samples,
                all_labels,
                jnp.ones_like(all_labels, dtype=bool),
                log_input=False,
                reduction="mean"
            )
            del state
            test_metrics = common_utils.get_metrics(test_metrics)
            test_metrics = jax.tree_util.tree_map(
                lambda e: e.sum(0), test_metrics)
            for k, v in test_metrics.items():
                if "count" in k:
                    continue
                test_metrics[k] /= test_metrics["count"]
            del test_metrics["count"]
            test_metrics["ece"] = ece
            test_metrics["cece"] = cece

            for k, v in test_metrics.items():
                if total_metrics.get(k) is None:
                    total_metrics[k] = [v]
                else:
                    total_metrics[k].append(v)
        for comb in comb_list:
            print(",".join(comb))
        columns = ["acc", "nll", "bs", "ece", "cnll", "cbs", "cece"]
        for k in columns:
            v = total_metrics[k]
            print(
                f"--------------------{k}--------------------")
            for c, ele in enumerate(v):
                print(f"{ele:.5f}")
            arr = jnp.stack(v)
            print("-------")
            print(f"{float(arr.mean()):.5f}")
            print(f"{float(arr.std()):.5f}")
        arr = []
        for k in columns:
            v = total_metrics[k]
            arr.append(v)
        arr = jnp.array(arr)
        arr = rearrange(arr, "m v -> v m")
        for a in arr:
            print(" ".join([f"{ele:.5f}" for ele in a]))

    def eval_dbn(checkpoint_list):
        # checkpoint_list is dictionary {"t235": ["1","2","3"], "t279": ["1","2","3"]}
        total_metrics = dict()
        comb_list = []
        N = len(list(checkpoint_list.values())[0])
        for idx in range(N):
            sub_rng = jax.random.fold_in(rng, idx)
            comb = [idx for _ in range(args.ensemble)]
            if "r2" in args.method:
                params_list = []
                stats_list = []
                for m in range(len(args.target_checkpoints)):
                    params, batch_stats, _ = load_ckpt(
                        args.target_checkpoints[m])
                    params_list.append(params)
                    stats_list.append(batch_stats)
                target_state = dict(
                    params=jax_utils.replicate(params_list),
                    batch_stats=jax_utils.replicate(stats_list)
                )
            params_list = []
            stats_list = []
            dir_list = []
            for m, ckpt_list in enumerate(checkpoint_list.values()):
                if m == args.ensemble:
                    break
                params, batch_stats, _ = load_ckpt(ckpt_list[comb[m]])
                params_list.append(params)
                stats_list.append(batch_stats)
                dir_list.append(ckpt_list[comb[m]].split("/")[-1])
            comb_list.append(dir_list)
            state = dict(
                params=jax_utils.replicate(params_list),
                batch_stats=jax_utils.replicate(stats_list)
            )
            # get temperature
            valid_loader = dataloaders["val_loader"](rng=None)
            valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
            all_samples = []
            all_labels = []
            for batch_idx, batch in enumerate(tqdm(valid_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                samples = step_sample(state, batch)
                _samples, _labels = get_batch_only(samples, batch)
                all_samples.append(_samples)
                all_labels.append(_labels)
            all_samples = jnp.concatenate(all_samples, axis=0)
            all_labels = jnp.concatenate(all_labels, axis=0)
            optimal_temp = get_optimal_temperature(all_samples, all_labels, log_input=False)
            optimal_temp = jax_utils.replicate(optimal_temp)
            # test
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            all_samples = []
            all_labels = []
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                batch_rng = jax.random.fold_in(sub_rng, batch_idx)
                state["rng"] = jax_utils.replicate(batch_rng)
                if "r2" in args.method:
                    batch = get_target(target_state, batch)
                metrics = step_test(state, batch, optimal_temp)
                test_metrics.append(metrics)

                samples = step_sample(state, batch)
                _samples, _labels = get_batch_only(samples, batch)
                all_samples.append(_samples)
                all_labels.append(_labels)
            all_samples = jnp.concatenate(all_samples, axis=0)
            all_labels = jnp.concatenate(all_labels, axis=0)
            all_cal_samples = jax.nn.softmax(
                jnp.log(all_samples)/jax_utils.unreplicate(optimal_temp),
                axis=-1
            )
            ece = evaluate_ece(
                all_samples,
                all_labels,
                jnp.ones_like(all_labels, dtype=bool),
                log_input=False,
                reduction="mean"
            )
            cece = evaluate_ece(
                all_cal_samples,
                all_labels,
                jnp.ones_like(all_labels, dtype=bool),
                log_input=False,
                reduction="mean"
            )

            del state
            test_metrics = common_utils.get_metrics(test_metrics)
            test_metrics = jax.tree_util.tree_map(
                lambda e: e.sum(0), test_metrics)
            for k, v in test_metrics.items():
                if "count" in k:
                    continue
                test_metrics[k] /= test_metrics["count"]
            del test_metrics["count"]
            test_metrics["ece"] = ece
            test_metrics["cece"] = cece

            for k, v in test_metrics.items():
                if total_metrics.get(k) is None:
                    total_metrics[k] = [v]
                else:
                    total_metrics[k].append(v)
        for comb in comb_list:
            print(",".join(comb))
        ignore = False
        columns = ["acc", "nll", "bs", "ece", "cnll", "cbs", "cece"]
        for k in columns:
            v = total_metrics[k]
            print(
                f"--------------------{k}--------------------")
            for c, ele in enumerate(v):
                if len(ele.shape) == 0:
                    print(f"{ele:.5f}")
                else:
                    print(ele)
                    ignore = True
            arr = jnp.stack(v)
            print("-------")
            print(f"{float(arr.mean()):.5f}")
            print(f"{float(arr.std()):.5f}")
        if "r2" in args.method:
            print("R2", (total_metrics["SSE"][0]/total_metrics["SST"][0]).mean())
        if not ignore:
            arr = []
            for k in columns:
                v = total_metrics[k]
                arr.append(v)
            arr = jnp.array(arr)
            arr = rearrange(arr, "m v -> v m")
            for a in arr:
                print(" ".join([f"{ele:.5f}" for ele in a]))

    def plot_dbn(checkpoint_list):
        def compare_sample(obj, new_obj):
            old_probs = obj["probs"]
            old_labels = obj["labels"]
            old_images = obj["images"]
            old_scores = obj["scores"]
            new_probs = rearrange(new_obj["probs"], "p b t d -> (p b) t d")
            new_labels = rearrange(new_obj["labels"], "p b -> (p b)")
            new_images = rearrange(
                new_obj["images"], "p b h w d -> (p b) h w d")
            new_score = 0
            new_score += ((new_probs[:, -1]-new_probs[:, -2])**2).sum(-1)
            new_score += ((new_probs[:, 0]-new_probs[:, 1])**2).sum(-1)
            # new_score += -jnp.std(new_probs[:, 0], axis=-1)
            # new_score += -jnp.std(new_probs[:, -1], axis=-1)
            # new_score += -((new_probs[:, 0]-new_probs[:, -1])**2).sum(-1)
            new_score_idx = jnp.argmin(new_score)
            if jnp.any(old_scores[0] > new_score[new_score_idx]):
                obj["probs"][0] = new_probs[new_score_idx, ...]
                obj["labels"][0] = new_labels[new_score_idx]
                obj["images"][0] = new_images[new_score_idx, ...]
                obj["scores"][0] = new_score[new_score_idx]
            elif jnp.any(old_scores[1] > new_score[new_score_idx]):
                obj["probs"][1] = new_probs[new_score_idx, ...]
                obj["labels"][1] = new_labels[new_score_idx]
                obj["images"][1] = new_images[new_score_idx, ...]
                obj["scores"][1] = new_score[new_score_idx]
            return obj

        idx = 0
        sub_rng = jax.random.fold_in(rng, idx)
        comb = jax.random.choice(sub_rng, len(
            checkpoint_list), shape=(args.ensemble,), replace=False)
        params_list = []
        stats_list = []
        dir_list = []
        # ensemble target
        for m in range(args.ensemble):
            params, batch_stats, _ = load_ckpt(checkpoint_list[comb[m]])
            params_list.append(params)
            stats_list.append(batch_stats)
            dir_list.append(checkpoint_list[comb[m]].split("/")[-1])
        state = dict(
            params=jax_utils.replicate(params_list),
            batch_stats=jax_utils.replicate(stats_list)
        )
        # DBN
        params, batch_stats, _ = load_ckpt(args.checkpoint)
        dbn_state = dict(
            params=jax_utils.replicate(params),
            batch_stats=jax_utils.replicate(batch_stats)
        )
        test_loader = dataloaders["tst_loader"](rng=None)
        test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
        obj = dict(
            probs=[None]*2,
            labels=[None]*2,
            images=[None]*2,
            scores=[float("inf")]*2
        )
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            batch_rng = jax.random.fold_in(sub_rng, batch_idx)
            dbn_state["rng"] = jax_utils.replicate(batch_rng)
            state["rng"] = jax_utils.replicate(batch_rng)
            target_probs = jax.pmap(test_de)(state, batch)
            probs = get_sample(dbn_state, batch)
            # target probs (p, b, d)
            assert len(target_probs.shape) == 3
            # new obj probs (p, b, t, d)
            assert len(probs.shape) == 4
            probs = jnp.concatenate(
                [probs, target_probs[:, :, None, :]], axis=2)
            new_obj = dict(
                probs=probs,
                labels=batch["labels"],
                images=batch["images"]
            )
            if batch_idx <500:
                obj = compare_sample(obj, new_obj)
        dbn_plot(
            probs=obj["probs"], labels=obj["labels"], images=obj["images"])

    def intermediate_dbn(checkpoint_list):
        idx = 0
        sub_rng = jax.random.fold_in(rng, idx)
        comb = jax.random.choice(sub_rng, len(
            checkpoint_list), shape=(args.ensemble,), replace=False)
        params_list = []
        stats_list = []
        dir_list = []
        # ensemble target
        for m in range(args.ensemble):
            params, batch_stats, _ = load_ckpt(checkpoint_list[comb[m]])
            params_list.append(params)
            stats_list.append(batch_stats)
            dir_list.append(checkpoint_list[comb[m]].split("/")[-1])
        state = dict(
            params=jax_utils.replicate(params_list),
            batch_stats=jax_utils.replicate(stats_list)
        )
        # DBN
        params, batch_stats, _ = load_ckpt(args.checkpoint)
        dbn_state = dict(
            params=jax_utils.replicate(params),
            batch_stats=jax_utils.replicate(batch_stats)
        )
        test_loader = dataloaders["tst_loader"](rng=None)
        test_loader = jax_utils.prefetch_to_device(test_loader, size=2)

        count = 0
        nll_sum = 0
        acc_sum = 0
        # target_nll_sum = 0
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            batch_rng = jax.random.fold_in(sub_rng, batch_idx)
            dbn_state["rng"] = jax_utils.replicate(batch_rng)
            state["rng"] = jax_utils.replicate(batch_rng)
            target_probs = jax.pmap(test_de)(state, batch) # (p,128//p,10)
            probs = get_sample(dbn_state, batch) # (p,128//p,6,10)
            probs = jnp.concatenate([probs, target_probs[:,:,None,:]], axis=-2)
            # shape_target_probs = target_probs.shape
            shape_probs = probs.shape
            # _target_probs = target_probs.reshape(-1, shape_target_probs[-1])
            _probs = probs.reshape(-1, shape_probs[-2], shape_probs[-1])
            assert len(batch["labels"].shape) == 2
            labels = batch["labels"].reshape(-1)
            marker = batch["marker"].reshape(-1,1)
            count += jnp.sum(marker)
            # target_nll = evaluate_nll_nrank(
            #     _target_probs, labels,log_input=False, reduction="none")
            # target_nll_sum += target_nll.sum(0)
            nll = _evaluate_nll_3rank(
                _probs, labels,log_input=False, reduction="none")
            nll = jnp.where(
                marker,
                nll,
                0
            )
            nll_sum += nll.sum(0)
            acc = _evaluate_acc_3rank(
                _probs, labels,log_input=False, reduction="none")
            acc = jnp.where(
                marker,
                acc,
                0
            )
            acc_sum += acc.sum(0)
        nll_mean = nll_sum/count
        acc_mean = acc_sum/count
        # target_mean = target_nll_sum/count
        print("nll")
        for n in nll_mean:
            print(f"{n:.4f}")
        print("acc")
        for n in acc_mean:
            print(f"{n*100:.2f}")
        # print(target_mean)


    if args.method in ["de"]:
        eval_de(checkpoint_list)
    elif args.method in ["dbn", "dbn-r2"]:
        eval_dbn(checkpoint_list)
    elif args.method == "dbn-plot":
        plot_dbn(checkpoint_list)
    elif args.method == "dbn-inter":
        intermediate_dbn(checkpoint_list)
    elif args.method in ["naive_ed", "proxy_end2"]:
        eval(params, batch_stats, checkpoint_list)
    elif args.method in ["mixup-shift"]:
        eval_mixupshift(params, batch_stats,checkpoint_list)


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
    parser.add_argument("--seed", default=1242300, type=int)
    parser.add_argument("--method", default="dbn", type=str)
    parser.add_argument("--corrupted", action="store_true")
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
