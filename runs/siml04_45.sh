echo 1
taskset -c 48-71 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --dsb_continuous
echo 2
