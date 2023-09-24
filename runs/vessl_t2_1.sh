echo 0
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --start_temp 2
echo 1
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --start_temp 3
echo 2