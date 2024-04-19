echo 0
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/t200/t3913/230925/v1110/0 \
    --save ./checkpoints/dbn/t200/t3913dist/230925/v1110/0 \
    --tag AtoABC2 \
    --seed 538825
echo 1
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/t200/t3913/230925/v1110/1 \
    --save ./checkpoints/dbn/t200/t3913dist/230925/v1110/1 \
    --tag AtoABC2 \
    --seed 379376
echo 2
python dbn_tidy.py \
    --config config_dsb/t200_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/t200/t3913/230925/v1110/2 \
    --save ./checkpoints/dbn/t200/t3913dist/230925/v1110/2 \
    --tag AtoABC2 \
    --seed 152285
echo 3
python proxy_end2.py \
    --config config_proxy_end2/t200_frnrelu_t5.yaml \
    --save ./checkpoints/proxy_end2/t200/t357913/230927/0 \
    --seed 991516
echo 4
python proxy_end2.py \
    --config config_proxy_end2/t200_frnrelu_t5.yaml \
    --save ./checkpoints/proxy_end2/t200/t357913/230927/1 \
    --seed 872963
echo 5
python proxy_end2.py \
    --config config_proxy_end2/t200_frnrelu_t5.yaml \
    --save ./checkpoints/proxy_end2/t200/t357913/230927/2 \
    --seed 569822
echo 6
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c10/t21113/230927/v118/1 \
    --tag AtoABC3 \
    --seed 29225
echo 7
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c10/t21113/230927/v118/1 \
    --save ./checkpoints/dbn/c10/t21113dist/230927/v118/1 \
    --tag AtoABC3 \
    --seed 77781
