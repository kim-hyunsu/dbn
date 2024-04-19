echo 1
taskset -c 0-47 python dbn_inputperturbation.py \
    --config config_dsb/c10_frnrelu_AtoABC_InputPerturbation.yaml \
    --save ./checkpoints/dbn/c10/inputperturbation0p1new/0 \
    --mixup_alpha 0.1
echo 2