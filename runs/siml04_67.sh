echo 0
CUDA_VISIBLE_DEVICES=6,7 taskset -c 72-95 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --distribution 3
echo 1
export CUDA_VISIBLE_DEVICES=6,7
taskset -c 72-95 python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 3812683646 \
    --save ./checkpoints/naive_ed/t200/230922/2
echo 2
taskset -c 72-95 python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 2091147107 \
    --save ./checkpoints/naive_ed/t200/230922/3
echo 3
taskset -c 72-95 python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 1990270746 \
    --save ./checkpoints/naive_ed/t200/230922/4
echo 4