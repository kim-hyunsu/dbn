echo 0
CUDA_VISIBLE_DEVICES=6,7 taskset -c 72-95 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml
echo 1