echo 0
taskset -c 0-47 python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/t200/t235/231113/v1110/1 \
    --lr 0.0005 \
    --seed 68265
echo 1