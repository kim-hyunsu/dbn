echo 0
python dbn_tidy.py \
    --config config_dsb/vi1000_frnrelu_AtoABC_ensemble.yaml \
    --seed 2916 \
    --save ./checkpoints/dbn/vi1000/50beta1_50beta2 \
    --beta1 0.005 \
    --beta2 0.005
echo 1
