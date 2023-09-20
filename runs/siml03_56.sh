echo 0
taskset -c 0-15 python sgd.py \
    --config ./config_sgd/c10_frnrelu_sgd.yaml \
    --nowandb > resnet_layers.log
echo 1
