echo 0
python sgd.py \
    --config config_sgd/c100_frnrelu_sgd.yaml \
    --seed 11 \
    --save ./checkpoints/frn100_sd11_be
echo 1
python sgd.py \
    --config config_sgd/c100_frnrelu_sgd.yaml \
    --seed 13 \
    --save ./checkpoints/frn100_sd13_be
echo 2
python sgd.py \
    --config config_sgd/c100_frnrelu_sgd.yaml \
    --seed 15 \
    --save ./checkpoints/frn100_sd15_be
echo 3
python sgd.py \
    --config config_sgd/c100_frnrelu_sgd.yaml \
    --seed 17 \
    --save ./checkpoints/frn100_sd17_be
echo 4
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --seed 3392819863 \
    --save ./checkpoints/c10/t235/230919/0
echo 5
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --seed 3947328007 \
    --save ./checkpoints/c10/t235/230919/1
echo 6
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --seed 377778458 \
    --save ./checkpoints/c10/t235/230919/2
echo 7
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --seed 1384393619 \
    --save ./checkpoints/c10/t235/230919/3
echo 8
python naive_ed.py \
    --config config_naive_ed/c10_frnrelu_shX_t3.yaml \
    --seed 2889929131 \
    --save ./checkpoints/c10/t235/230919/4
echo 9