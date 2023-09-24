export CUDA_VISIBLE_DEVICES=4,5,6,7
echo 0
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t5.yaml \
    --seed 2249972020 \
    --save ./checkpoints/naive_ed/c10/t23579/230923/0
echo 1
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t5.yaml \
    --seed 497488857 \
    --save ./checkpoints/naive_ed/c10/t23579/230923/1
echo 2
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t5.yaml \
    --seed 2696417285 \
    --save ./checkpoints/naive_ed/c10/t23579/230923/2
echo 3
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t5.yaml \
    --seed 1349922039 \
    --save ./checkpoints/naive_ed/c10/t23579/230923/3
echo 4
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t5.yaml \
    --seed 2910230587 \
    --save ./checkpoints/naive_ed/c10/t23579/230923/4
echo 5
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t7.yaml \
    --seed 3914743490 \
    --save ./checkpoints/naive_ed/c10/t235791113/230923/0
echo 6
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t7.yaml \
    --seed 1587805547 \
    --save ./checkpoints/naive_ed/c10/t235791113/230923/1
echo 7
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t7.yaml \
    --seed 377502791 \
    --save ./checkpoints/naive_ed/c10/t235791113/230923/2
echo 8
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t7.yaml \
    --seed 2063208277 \
    --save ./checkpoints/naive_ed/c10/t235791113/230923/3
echo 9
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t7.yaml \
    --seed 761055307 \
    --save ./checkpoints/naive_ed/c10/t235791113/230923/4
echo 10
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t9.yaml \
    --seed 179086698 \
    --save ./checkpoints/naive_ed/c10/t2357911131517/230923/0
echo 11
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t9.yaml \
    --seed 1170717227 \
    --save ./checkpoints/naive_ed/c10/t2357911131517/230923/1
echo 12
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t9.yaml \
    --seed 2306825241 \
    --save ./checkpoints/naive_ed/c10/t2357911131517/230923/2
echo 13
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t9.yaml \
    --seed 757026008 \
    --save ./checkpoints/naive_ed/c10/t2357911131517/230923/3
echo 14
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_t9.yaml \
    --seed 2178467949 \
    --save ./checkpoints/naive_ed/c10/t2357911131517/230923/4
echo 15
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t5.yaml \
    --seed 2249972020 \
    --save ./checkpoints/naive_ed/c100/t11131523/230923/0
echo 16
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t5.yaml \
    --seed 497488857 \
    --save ./checkpoints/naive_ed/c100/t11131523/230923/1
echo 17
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t5.yaml \
    --seed 2696417285 \
    --save ./checkpoints/naive_ed/c100/t11131523/230923/2
echo 18
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t5.yaml \
    --seed 1349922039 \
    --save ./checkpoints/naive_ed/c100/t11131523/230923/3
echo 19
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t5.yaml \
    --seed 2910230587 \
    --save ./checkpoints/naive_ed/c100/t11131523/230923/4
echo 20
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t7.yaml \
    --seed 3914743490 \
    --save ./checkpoints/naive_ed/c100/t1113152357/230923/0
echo 21
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t7.yaml \
    --seed 1587805547 \
    --save ./checkpoints/naive_ed/c100/t1113152357/230923/1
echo 22
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t7.yaml \
    --seed 377502791 \
    --save ./checkpoints/naive_ed/c100/t1113152357/230923/2
echo 23
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t7.yaml \
    --seed 2063208277 \
    --save ./checkpoints/naive_ed/c100/t1113152357/230923/3
echo 24
taskset -c 48-95 python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_t7.yaml \
    --seed 761055307 \
    --save ./checkpoints/naive_ed/c100/t1113152357/230923/4
echo 25