echo 0
taskset -c 48-71 python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --resume ./checkpoints/dbn/t200/t235/231113/v1110/7 \
    --beta2 0.005 \
    --beta1 0.005 \
    --optim_lr 0.0005 \
    --seed 68265
echo 1
taskset -c 48-71 python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/t200/t235/231113/v1110/7 \
    --save ./checkpoints/dbn/t200/t235dist/231113/v1110/7 \
    --beta2 0.005 \
    --beta1 0.005 \
    --optim_lr 0.0005 \
    --seed 6865
echo 2