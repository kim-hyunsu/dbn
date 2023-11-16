echo 0
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/t200/t279/231114/v1110/1 \
    --tag AtoABC2 \
    --beta2 0.001 \
    --beta1 0.001 \
    --optim_lr 0.00075 \
    --seed 68
echo 1
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --distill ./checkpoints/dbn/t200/t279/231114/v1110/1 \
    --save ./checkpoints/dbn/t200/t279dist/231114/v1110/1 \
    --tag AtoABC2 \
    --beta2 0.001 \
    --beta1 0.001 \
    --optim_lr 0.00075 \
    --optim_ne 100 \
    --seed 80101
echo 2