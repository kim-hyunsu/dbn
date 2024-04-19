# Fast Ensembling with Diffusion Schrödinger Bridge

Official implementation of _Fast Ensembling with Diffusion Schrödinger Bridge_ (ICLR 2024).

[[`ICLR`](https://openreview.net/forum?id=Mgq6kxl115)]

## Installation

```bash
source scripts/requirements.sh
```

## Collect checkpoints to distill
```bash
python sgd.py
```

## DBN
```bash
python dbn_tidy.py
```

## Ensemble Distillation
```bash
python naive_ed.py
```

## Proxy Ensemble Distribution Distillation
```bash
python proxy_end2.py
```

## Evaluate trained models
```bash
python test_dbn.py
```

## Citation

```
@inproceedings{kim2024fast,
    title     = {Fast Ensembling with Diffusion Schrodinger Bridge},
    author    = {Kim, Hyunsu and Yoon, Jongmin and Lee, Juho},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2024},
}
```