echo 0
python dbn_inputperturbation.py \
    --config config_dsb/c10_frnrelu_AtoABC_InputPerturbation.yaml \
    --save ./checkpoints/dbn/c10/inputperturbation0p7new/1 \
    --mixup_alpha 0.7
echo 1