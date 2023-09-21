echo 0
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --seed 11 \
    --save ./checkpoints/frn200_sd11_be --nowandb
echo 1
