# Fast Ensembling with Diffusion Schrödinger Bridge

Official implementation of _Fast Ensembling with Diffusion Schrödinger Bridge_ [[`ICLR 2024`](https://openreview.net/forum?id=Mgq6kxl115)].



## Installation (CUDA 11.7)
### Conda
```bash
conda create -n dbn python=3.9.16
conda activate dbn
```
### pip packages
```bash
source scripts/requirements.sh
```

## Install datasets
### CIFAR10
```bash
cd data
wget https://www.dropbox.com/s/8s5unpaxdrt6r7e/CIFAR10_HMC.tar.gz
tar -xvzf CIFAR10_HMC.tar.gz
mv CIFAR10_HMC CIFAR10_x32
cd -
```
### CIFAR100
```bash
cd data
wget https://www.dropbox.com/s/bvljrqsttemfdzv/CIFAR100_HMC.tar.gz
tar -xvzf CIFAR100_HMC.tar.gz
mv CIFAR100_HMC CIFAR100_x32
cd -
```
### TinyImageNet
```bash
cd data
wget https://www.dropbox.com/s/rcxylz8dn7fm03m/TinyImageNet200_x32.tar.gz
tar -xvzf TinyImageNet200_x32.tar.gz
cd -
```
### ImageNet (64x64)
```bash
cd data
wget https://www.dropbox.com/s/urh77zy42xxzhgo/ImageNet1k_x64.tar.gz
tar -xvzf ImageNet1k_x64.tar.gz
cd -
```

## Collect checkpoints to distill
```bash
python sgd.py \
    --config config_sgd/c10_frnrelu_sgd.yaml \
    --save ./checkpoints/frn_sd2024_be.yaml \
    --seed 2024
```

## Register the trained checkpoints
You need to modify utils.py
```python
def model_list(data_name, model_style, shared_head=False, tag=""):
    ...
    elif data_name == "CIFAR10_x32" and model_style == "FRN-Swish":
        ...
        elif tag == "AtoABC":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd5_be",
                "./checkpoints/frn_sd2024_be",
            ]
        ...
```

## DBN training (example checkpoints are already given)
```bash
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/example
```

## DBN diffusion step distillation
```bash
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/example \
    --save ./checkpoints/dbn/c10/example_distilled
```

## Ensemble Distillation
```bash
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --save ./checkpoints/naive_ed/c10/example
```

## Ensemble Distribution Distillation
```bash
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml \
    --save ./checkpoints/proxy_end2/c10/example
```

## Evaluate trained models

### DE
```bash
python test_dbn.py \
    --config config_test/c10/de_example.yaml
```
### DBN
```bash
python test_dbn.py \
    --config config_test/c10/dbn_example.yaml
```
### ED
```bash
python test_dbn.py \
    --config config_test/c10/naive_ed_example.yaml
```
### END^2
```bash
python test_dbn.py \
    --config config_test/c10/proxy_end2_example.yaml
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