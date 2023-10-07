echo 0
taskset -c 48-71 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t279/230925/3 \
    --version v1.1.8 \
    --centering \
    --tag AtoABC2 \
    --seed 94743838
echo 1