echo 0
taskset -c 0-47 python sgd.py \
    --config config_sgd/t200_frnrelu_sgd.yaml \
    --save ./checkpoints/frn200_sd2_90K \
    --seed 2
echo 1