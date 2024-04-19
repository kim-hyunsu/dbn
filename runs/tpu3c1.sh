echo 0
python dbn_tidy.py \
    --config config_dsb/vi1000_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/vi1000/50beta1_50beta2 \
    --save ./checkpoints/dbn/vi1000/2dist \
    --seed 86868
echo 1
