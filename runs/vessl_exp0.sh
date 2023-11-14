echo 0
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/t200/t235/231113/v1110/0 \
    --optim_lr 0.00075 \
    --seed 68265
echo 1