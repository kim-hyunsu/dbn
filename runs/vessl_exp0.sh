echo 0
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --seed 3 \
    --save ./checkpoints/frn200_sd3_be
echo 1
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --seed 5 \
    --save ./checkpoints/frn200_sd5_be
echo 2
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --seed 7 \
    --save ./checkpoints/frn200_sd7_be
echo 3
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --seed 9 \
    --save ./checkpoints/frn200_sd9_be
echo 4