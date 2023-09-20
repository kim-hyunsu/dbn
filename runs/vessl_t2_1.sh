echo 0
taskset -c 48-71 python dbn.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml --nowandb
echo 1
