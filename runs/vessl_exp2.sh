echo 0
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/t200/t235/231113/v1110/5 \
    --beta2 0.005 \
    --beta1 0.005 \
    --seed 68265
echo 1