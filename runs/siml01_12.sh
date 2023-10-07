echo 0
taskset -c 16-31 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t235/230925/v117/0
echo 1
