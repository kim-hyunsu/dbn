echo 0
python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --save ./checkpoints/frn200_sd2_90K_nobias \
    --model_nobias \
    --seed 2
echo 1