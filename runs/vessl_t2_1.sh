echo 0
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --distill ./checkpoints/dbn/c10/t235/230925/v117/0 \
    --distill_alpha 0.5 \
    --save ./checkpoints/dbn/c10/t235dist/230925/v117/0
echo 1