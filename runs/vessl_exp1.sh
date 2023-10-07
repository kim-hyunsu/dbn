echo 0
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_t5.yaml \
    --save ./checkpoints/naive_ed/t200/t357913/230929/0 \
    --seed 872963
echo 1
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_t5.yaml \
    --save ./checkpoints/naive_ed/t200/t357913/230929/1 \
    --seed 8729
echo 2