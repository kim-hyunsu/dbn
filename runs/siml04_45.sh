echo 0
taskset -c 48-71 python dbn.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315
echo 1
