export CUDA_VISIBLE_DEVICES=4,5
echo 0
taskset -c 48-71 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --distribution 1 \
    --mixup_alpha 0.1
echo 1
taskset -c 48-71 python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 2249972020 \
    --save ./checkpoints/naive_ed/t200/230922/0
echo 2
taskset -c 48-71 python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 1818705916 \
    --save ./checkpoints/naive_ed/t200/230922/1
echo 3
