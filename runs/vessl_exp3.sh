echo 16
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t235/231111/v118_8temp/2 \
    --start_temp 8 \
    --seed 682651
echo 17
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t235/231111/v118_8temp/2 \
    --save ./checkpoints/dbn/c10/t235dist/231111/v118_8temp/2 \
    --start_temp 8 \
    --seed 175942
echo 18