echo 0
python dbn_tidy.py \
    --config config_dsb/vi1000_frnrelu_AtoABC_distill.yaml \
    --seed 2711 \
    --distill ./checkpoints/dbn/vi1000/t235/231113/v1113/0 \
    --save ./checkpoints/dbn/vi1000/fancy-oathdist \
    --beta1 0.001 \
    --beta2 0.001 \
    --optim_lr 0.00075
echo 1
