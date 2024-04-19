echo 0
python dbn_tidy.py \
    --config config_dsb/vi1000_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/vi1000/10beta1_10beta2 \
    --save ./checkpoints/dbn/vi1000/1dist \
    --seed 99999
echo 1
