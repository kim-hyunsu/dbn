echo 0
python dbn_inputperturbation.py \
    --config config_dsb/c10_frnrelu_AtoABC_InputPerturbation.yaml \
    --save ./checkpoints/dbn/c10/inputperturbation0p9new/1 \
    --mixup_alpha 0.9
echo 1