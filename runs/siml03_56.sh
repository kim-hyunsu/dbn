echo 0
taskset -c 72-95 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t21113/v118 \
    --tag AtoABC3 \
    --seed 786871
echo 1
taskset -c 72-95 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --distill ./checkpoints/dbn/c10/t21113/v118/prime-lake-1060 \
    --distill_alpha 0.5 \
    --save ./checkpoints/dbn/c10/t21113dist/v118 \
    --tag AtoABC3 \
    --seed 3188656917
echo 2