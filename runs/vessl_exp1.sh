echo 0
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 2249972020 \
    --save ./checkpoints/naive_ed/t200/230922/0
echo 1
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 1818705916 \
    --save ./checkpoints/naive_ed/t200/230922/1
echo 2
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 3812683646 \
    --save ./checkpoints/naive_ed/t200/230922/2
echo 2
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 2091147107 \
    --save ./checkpoints/naive_ed/t200/230922/3
echo 3
python naive_ed.py \
    --config config_naive_ed/t200_frnrelu_shX_t3.yaml \
    --seed 1990270746 \
    --save ./checkpoints/naive_ed/t200/230922/4
echo 4