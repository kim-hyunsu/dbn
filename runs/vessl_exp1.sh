echo 0
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --version v1.1.8 \
    --start_cls 300
echo 1
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --version v1.1.8 \
    --start_cls 500
echo 2
