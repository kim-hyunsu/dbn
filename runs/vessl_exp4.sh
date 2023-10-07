echo 0
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t235/230927/v118/1 \
    --seed 279856
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t235/230927/v118/1 \
    --save ./checkpoints/dbn/c10/t235dist/230927/v118/1 \
    --seed 129101
echo 2
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t235/230927/v118/2 \
    --seed 191797
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t235/230927/v118/2 \
    --save ./checkpoints/dbn/c10/t235dist/230927/v118/2 \
    --seed 37332
echo 2
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t279/230927/v118/1 \
    --tag AtoABC2 \
    --seed 79121
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t279/230927/v118/1 \
    --save ./checkpoints/dbn/c10/t279dist/230927/v118/1 \
    --tag AtoABC2 \
    --seed 19
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t279/230927/v118/2 \
    --tag AtoABC2 \
    --seed 134155
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t279/230927/v118/2 \
    --save ./checkpoints/dbn/c10/t279dist/230927/v118/2 \
    --tag AtoABC2 \
    --seed 29225
echo 1