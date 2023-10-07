echo 0
taskset -c 48-63 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t235/230925/v117/1 \
    --seed 1012778689
echo 1
